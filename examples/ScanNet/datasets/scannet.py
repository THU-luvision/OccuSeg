import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
from functools import partial
import torch.nn.functional as F
import logging
from sklearn.neighbors import KDTree
import pdb
from torch_scatter import scatter_mean,scatter_std,scatter_add,scatter_max

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training logger')
logger.setLevel(logging.DEBUG)




class ScanNet(object):
    def __init__(self,
                 train_pth_path,
                 val_pth_path,
                 config):
        if isinstance(train_pth_path,list):
            self.train_pths = []
            for train_pth in train_pth_path:
                self.train_pths += glob.glob(train_pth)
        else:
            self.train_pths = glob.glob(train_pth_path)
        self.val_pths = glob.glob(val_pth_path)
        self.train, self.val = [], []

        self.blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        self.blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        self.blur2 = np.ones((1, 1, 3)).astype('float32') / 3
        self.scale = config['scale']
        self.val_reps = config['val_reps']
        self.batch_size = config['batch_size']
        self.dimension = config['dimension']
        self.full_scale = config['full_scale']
        self.use_normal = config['use_normal']
        self.use_elastic = config['use_elastic']
        self.use_feature = config['use_feature']
        self.use_rotation_noise = config['use_rotation_noise']
        self.regress_sigma = config['regress_sigma']
        self.PRINT_ONCE_FLAG = 0
        torch.manual_seed(100)  # cpu
        torch.cuda.manual_seed(100)  # gpu
        np.random.seed(100)  # numpy
        torch.backends.cudnn.deterministic = True  # cudnn

    def elastic(self, x, gran, mag):
        if not self.use_elastic:
            return x

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, self.blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, self.blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, self.blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, self.blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, self.blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, self.blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def trainMerge(self, tbl, train):
        locs = []
        feats = []
        labels = []
        normals = []
        pth_files = []
        index_list = []
        totalPoints = 0
        sizes = []
        masks = []
        offsets = []
        displacements = []
        regions = []
        region_masks = []
        region_indexs = []
        instance_masks = []
        instance_sizes = []
        for idx, i in enumerate(tbl):

            a, b, c = train[i]['coords'], train[i]['colors'], train[i]['w']
            if 'normal' in train[i]:
                d = train[i]['normals']
            else:
                d = train[i]['coords']

            if ('depth' in train[i]):
                e = train[i]['depth']
            else:
                e = train[i]['coords']
            if('region' in train[i]):
                region_parts = train[i]['region']
            else:
                region_parts = c[:,1]
            pth_files.append(self.train_pths[i])
            # checked
            # logger.debug("CHECK RANDOM SEED(np seed): sample id {}".format(np.random.randn(3, 3)))
            m = np.eye(3)
            if (self.use_rotation_noise):
                m = m + np.random.randn(3, 3) * 0.1
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m *= self.scale
            #            m *= (1 + np.random.randn(1) * 0.2) # add scale distortion, which might be useful?
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

            # align with origin
            # a = a - np.mean(a,0)
            """
            theta=np.random.rand()*2*math.pi * 0.05
            m=np.matmul(m,[[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
            theta=np.random.rand()*2*math.pi * 0.05
            m=np.matmul(m,[[1,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]])
            """
            a = np.matmul(a, m)
            d = np.matmul(d, m) / self.scale
            random_scale = np.random.rand()
            a = self.elastic(a, 6 * self.scale // 50, random_scale * 40 * self.scale / 50)
            random_scale = np.random.rand()
            a = self.elastic(a, 20 * self.scale // 50, random_scale * 160 * self.scale / 50)

            m = a.min(0)
            M = a.max(0)
            q = M - m
            #        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
            offset = (np.min(a[:, 0]) - 10, np.min(a[:, 1]) - 10, np.min(a[:, 2]) - 10) + np.random.rand(3)
            a = a - offset
            idxs = (a.min(1) >= 0) * (a.max(1) < self.full_scale)
            # random drop part of the point clouds

            """
            pre_total_points = 0
            post_total_points = 0
            if (np.sum(idxs) > 400000):
                pre_total_points = np.sum(idxs)
                idxs = idxs * ((a[:, 0] < np.median(a[:, 0])) > 0)
                post_total_points = np.sum(idxs)
            if (np.sum(idxs) > 400000):
                idxs = idxs * ((a[:, 1] < np.median(a[:, 1])) > 0)
                post_total_points = np.sum(idxs)
            if (np.sum(idxs) > 400000):
                filtered_a = a.copy()
                filtered_a = a[idxs]
                idxs = idxs * ((a[:, 0] < np.median(filtered_a[:, 0]))> 0)
                post_total_points = np.sum(idxs)
            if (np.sum(idxs) > 400000):
                filtered_a = a.copy()
                filtered_a = a[idxs]
                idxs = idxs * ((a[:, 1] < np.median(filtered_a[:, 1]))> 0)
                post_total_points = np.sum(idxs)
            """
            # randomly zoom out one class
#            idxs[c == (3+np.random.randint(14))] = 0
#            idxs[c == (3-np.random.randint(14))] = 0
            a = a[idxs]
            b = b[idxs]
            c = c[idxs].astype(np.int32)
            d = d[idxs]
            e = e[idxs]
            region_numpy = region_parts[idxs]
            region = torch.from_numpy(region_numpy)
            [region_index, region_mask] = np.unique(region_numpy, False, True)
            region_mask = torch.from_numpy(region_mask)
            region_index = torch.from_numpy(region_index)
            #        a=torch.from_numpy(a).long()


            # generate masks for each instance:
            c[:,1] = np.unique(c[:,1], False, True)[1]
            instance_mask = c[:,1]
            instance_size = scatter_add(torch.ones([a.shape[0]]), torch.Tensor(instance_mask).long(), dim = 0)
            instance_size = torch.gather(instance_size, dim = 0, index = torch.Tensor(instance_mask).long())
            mask = torch.zeros((a.shape[0], np.max(c[:,1]) + 1), dtype=torch.float32)
            mask[torch.arange(a.shape[0]), c[:,1].astype(np.int32)] = 1

            a = torch.from_numpy(a).float()
            e = torch.from_numpy(e).float()

            displacement = torch.zeros([a.shape[0],3], dtype = torch.float32)
            offset = torch.zeros(a.shape[0], dtype = torch.float32)
            for count in range(mask.shape[1]):
                indices = mask[:,count] == 1
#                cls = torch.from_numpy(c[:,0])[indices][0]
#                if(cls > 1):
#                    random_shift = (torch.rand(3)) * self.scale * 3 # randomly shift 3 meters
#                    random_shift[2] = 0
#                    a[indices,:] += random_shift
                mean = torch.mean(a[indices,:],dim = 0)
                distance = torch.norm(a[indices,:] - mean,dim = 1)
                offset[indices] = torch.exp(- (distance / self.scale/  self.regress_sigma ) ** 2 )
                displacement[indices,:] = (a[indices,:] - mean) / self.scale

            totalPoints = totalPoints + a.shape[0]
            #            if totalPoints < 1500000:
            if True:
                locs.append(torch.cat([a, torch.FloatTensor(a.shape[0], 1).fill_(idx)], 1))

                lf = a - torch.mean(a, dim = 0).view(1,-1).expand_as(a)
                l_feature = lf.div(torch.norm(lf, p=2, dim=1).view(-1,1).expand_as(lf))
                color = torch.from_numpy(b).float() + torch.randn(3).float() * 0.1
                color = torch.clamp(color, -1, 1)
                tmp_feature = []
                if 'l' in self.use_feature:
                    tmp_feature.append(l_feature)
                if 'c' in self.use_feature:
                    tmp_feature.append(color)
                if 'n' in self.use_feature:
                    tmp_feature.append(torch.from_numpy(d).float())
                if 'd' in self.use_feature:
                    tmp_feature.append(e)
                if 'h' in self.use_feature:
                    tmp_feature.append(a[:, 2:3])
                # concat in channel dim
                tmp_feature = torch.cat(tmp_feature, dim=1)
                feats.append(tmp_feature)
                sizes.append(torch.tensor(np.unique(c[:,1]).size))
                masks.append(mask)
                regions.append(region)
                region_masks.append(region_mask)
                region_indexs.append(region_index)
                labels.append(torch.from_numpy(c))
                normals.append(torch.from_numpy(d).float().cpu())
                offsets.append(offset)
                displacements.append(displacement)
                instance_masks.append(torch.Tensor(instance_mask))
                instance_sizes.append(instance_size)
                index_list.append(torch.from_numpy(idxs.astype(int)))
            else:
                print("lost file for training: ", self.train_pths[i])

        local_batch_size = len(locs)
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        labels = torch.cat(labels, 0)
        sizes = torch.stack(sizes, 0)
        normals = torch.cat(normals, 0)
        offsets = torch.cat(offsets,0)
        displacements = torch.cat(displacements, 0)
        instance_masks = torch.cat(instance_masks, 0)
        instance_sizes = torch.log(torch.cat(instance_sizes, 0))
        regions = torch.cat(regions,0)
        region_masks = torch.cat(region_masks, 0)
        region_indexs = torch.cat(region_indexs, 0)

        if not self.use_normal:
            normals = torch.zeros(3, 3).float()
        return {'x': [locs, feats, normals, local_batch_size], 'y': labels.long(), 'id': tbl, 'elastic_locs': a,
                'pth_file': pth_files,
                'idxs': index_list,
                'masks': masks,
                'instance_masks':instance_masks,
                'instance_sizes':instance_sizes,
                'sizes':sizes,
                'offsets': offsets.view(-1,1),
                'displacements':displacements,
                'regions':regions,
                'region_masks':region_masks,
                'region_indexs':region_indexs}

    def valMerge(self, tbl, val, valOffsets):
        locs = []
        feats = []
        labels = []
        point_ids = []
        pth_files = []
        index_list = []
        normals = []
        sizes = []
        masks = []
        offsets = []
        displacements = []
        regions = []
        region_masks = []
        region_indexs = []
        totalPoints = 0
        instance_masks = []
        instance_sizes = []
        for idx, i in enumerate(tbl):
            a, b, c = val[i]['coords'], val[i]['colors'], val[i]['w']
            if 'normal' in val[i]:
                d = val[i]['normals']
            else:
                d = val[i]['coords']

            if ('depth' in val[i]):
                e = val[i]['depth']
            else:
                e = val[i]['coords']

            if('region' in val[i]):
                region_parts = val[i]['region']
            else:
                region_parts = c[:,1]

            pth_files.append(self.val_pths[i])
            m = np.eye(3)
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m *= self.scale
            theta = np.random.rand() * 2 * math.pi
            #            theta = np.random.randint(4) * 0.5 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            # align with origin
            # a = a - np.mean(a,0)

            a = np.matmul(a, m) + self.full_scale / 2 + np.random.uniform(-2, 2, 3)
            d = np.matmul(d, m) / self.scale
            m = a.min(0)
            M = a.max(0)
            offset = (np.min(a[:, 0]) - 10, np.min(a[:, 1]) - 10, np.min(a[:, 2]) - 10) + np.random.rand(3)
            a -= offset
            #        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
            #        offset = ( np.min(a[:,0])-10,np.min(a[:,1])-10,np.min(a[:,2])-10) + np.random.rand(3)
            #        a = a - offset
            idxs = (a.min(1) >= 0) * (a.max(1) < self.full_scale)
            #        if(np.max(a[:,0]) > 2000 or np.max(a[:,1]) > 2000 or np.max(a[:,2]) > 1000):
            #            print("warning! scale too large!")

            a = a[idxs]
            b = b[idxs]
            c = c[idxs].astype(np.int32)
            d = d[idxs]
            e = e[idxs]
            region_numpy = region_parts[idxs]
            region = torch.from_numpy(region_numpy)
            [region_index, region_mask] = np.unique(region_numpy, False, True)
            region_mask = torch.from_numpy(region_mask)
            region_index = torch.from_numpy(region_index)

            c[:,1] = np.unique(c[:,1], False, True)[1]
            instance_mask = c[:,1]
            instance_size = scatter_add(torch.ones([a.shape[0]]), torch.Tensor(instance_mask).long(), dim = 0)
            instance_size = torch.gather(instance_size, dim = 0, index = torch.Tensor(instance_mask).long())

            mask = torch.zeros((a.shape[0], np.max(c[:,1]) + 1), dtype=torch.float32)
            mask[torch.arange(a.shape[0]), c[:,1]] = 1

            a = torch.from_numpy(a).float()
            e = torch.from_numpy(e).float()
            displacement = torch.zeros([a.shape[0],3], dtype = torch.float32)
            offset = torch.zeros(a.shape[0], dtype = torch.float32)
            for count in range(mask.shape[1]):
                indices = mask[:,count] == 1
                mean = torch.mean(a[indices,:],dim = 0)
                distance = torch.norm(a[indices,:] - mean,dim = 1)
                offset[indices] = torch.exp(- (distance / self.scale/ self.regress_sigma ) ** 2 )
                displacement[indices,:] = (a[indices,:] - mean) / self.scale

            totalPoints = totalPoints + a.shape[0]

            #            if totalPoints < 1500000:
            if True:
                locs.append(torch.cat([a, torch.FloatTensor(a.shape[0], 1).fill_(idx)], 1))

                lf = a - torch.mean(a, dim = 0).view(1,-1).expand_as(a)
                l_feature = lf.div(torch.norm(lf, p=2, dim=1).view(-1,1).expand_as(lf))
                color = torch.from_numpy(b).float() + torch.randn(3).float() * 0.1
                color = torch.clamp(color, -1, 1)
                tmp_feature = []
                if 'l' in self.use_feature:
                    tmp_feature.append(l_feature)
                if 'c' in self.use_feature:
                    tmp_feature.append(color)
                if 'n' in self.use_feature:
                    tmp_feature.append(torch.from_numpy(d).float())
                if 'd' in self.use_feature:
                    tmp_feature.append(e)
                if 'h' in self.use_feature:
                    tmp_feature.append(a[:, 2:3])
                # concat in channel dim
                tmp_feature = torch.cat(tmp_feature, dim=1)
                feats.append(tmp_feature.float())
                labels.append(torch.from_numpy(c))
                masks.append(mask)
                sizes.append(torch.tensor(np.unique(c[:,1]).size))
                normals.append(torch.from_numpy(d).float().cpu())
                point_ids.append(torch.from_numpy(np.nonzero(idxs)[0] + valOffsets[i]))
                index_list.append(torch.from_numpy(idxs.astype(int)))
                offsets.append(offset)
                displacements.append(displacement)
                regions.append(region)
                region_masks.append(region_mask)
                region_indexs.append(region_index)
                instance_sizes.append(instance_size)
                instance_masks.append(torch.Tensor(instance_mask))
            else:
                print("lost file for training: ", self.val_pths[i])

        local_batch_size = len(locs)
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        labels = torch.cat(labels, 0)
        sizes = torch.stack(sizes, 0)
        point_ids = torch.cat(point_ids, 0)
        normals = torch.cat(normals, 0)
        offsets = torch.cat(offsets,0)
        displacements = torch.cat(displacements,0)
        regions = torch.cat(regions,0)
        region_masks = torch.cat(region_masks,0)
        region_indexs = torch.cat(region_indexs,0)
        instance_masks = torch.cat(instance_masks, 0)
        instance_sizes = torch.log(torch.cat(instance_sizes, 0))
        if not self.use_normal:
            normals = torch.zeros(3, 3).float()


        return {'x': [locs, feats, normals, local_batch_size], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids,
                'pth_file': pth_files,
                'idxs': index_list,
                'masks': masks,
                'instance_masks':instance_masks,
                'instance_sizes':instance_sizes,
                'sizes':sizes,
                'offsets': offsets.view(-1,1),
                'displacements': displacements,
                'regions':regions,
                'region_masks':region_masks,
                'region_indexs':region_indexs}

    def load_data(self):
        for x in torch.utils.data.DataLoader(
                self.train_pths,
                collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
            self.train.append(x)
        for x in torch.utils.data.DataLoader(
                self.val_pths,
                collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
            self.val.append(x)
        print('Training examples:', len(self.train))
        print('Validation examples:', len(self.val))

        if len(self.train) == 0 or len(self.val) == 0:
            raise ValueError('Please prepare_data.py to generate training files')

        max_instance_train = 0
        for idx,x in enumerate(self.train):
            self.train[idx]['w'][:,1] = np.unique(x['w'][:,1], False, True)[1]
            max_instance_train = np.max([max_instance_train,np.max(self.train[idx]['w'][:,1]) ])
        self.max_instance_train = max_instance_train + 1
        train_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.train))), batch_size=self.batch_size,
            collate_fn=partial(self.trainMerge, train=self.train), num_workers=10, shuffle=True)

        valOffsets = [0]
        valLabels = []
        for idx, x in enumerate(self.val):
            self.val[idx]['w'][:,1] = np.unique(x['w'][:,1], False, True)[1]
            valOffsets.append(valOffsets[-1] + x['w'].shape[0])
            valLabels.append(x['w'][:,0].astype(np.int32))
            # d = x[2].astype(np.int32)
        valLabels = np.hstack(valLabels)

        val_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.val))), batch_size=self.batch_size,
            collate_fn=partial(self.valMerge, val=self.val, valOffsets=valOffsets), num_workers=10,
            shuffle=True)
        return valOffsets, train_data_loader, val_data_loader, valLabels
