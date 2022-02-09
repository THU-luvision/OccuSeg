- zyhseg000: 网络结构最多最复杂的原版网络
- zyhseg001: base网络结构只修改batch size 和voxel
> hint: 合理分配每个服务器中任务占用的资源
- zyhseg004: 找到了一个合适的lr decay policy 并跑一个baseline
- zyhseg005: 验证大卷积核是能够提高accuracy的
    - 01: batch:2 scale:50 kernel:3
    - 02: batch:2 scale:50 kernel:4
    - 03: batch:2 scale:50 kernel:5
    - 04: batch:4 scale:20 kernel:3
    - 05: batch:4 scale:20 kernel:4
    - 06: batch:4 scale:20 kernel:5
    - 10: batch:4 scale:50 kernel:3
    - 07: batch:4 scale:50 kernel:4
    - 08: batch:4 scale:50 kernel:5
    - 09: batch:8 scale:50 kernel:5

> 1. warning: add new branch of lhanaf's SCN, where invariant conv kernel and dailated conv kernel have been implemented.
> 2. remove category tp/fp/log
> 3. focus on with elastic first
- zyhseg006:寻找elastic和use_normal的对比
    - 10:elastic=1,use_normal=0
    - 11:elastic=1,use_normal=1
- zyhseg007:寻找elastic和use_normal的对比,在大量参数的情况下
    - m=
    - 00:elastic=0,use_normal=0
    - 01:elastic=0,use_normal=1
    - 10:elastic=1,use_normal=0
    - 11:elastic=1,use_normal=1
> wait to merge
> 1. prepare_data.py

------------------------------------------------
------------------------------------------------

- zyhseg012:在scale=20的情况下看input_feature的效果
------------------------------------------------
    - [x] 01: use normal=0, use elastic=1, use feature= c
    - [x] 01_1: use normal=0, use elastic=0, use feature= c
    - [ ] 02: use normal=0, use elastic=1, use feature= cd
    - [ ] 03: use normal=0, use elastic=1, use feature= ch
    - [ ] 04: use normal=0, use elastic=1, use feature= cn
------------------------------------------------
    - [x] 05: use normal=1, use elastic=1, use feature= c
    - [x] 05_1: use normal=1, use elastic=0, use feature= c
    - [ ] 06: use normal=1, use elastic=1, use feature= cd
    - [ ] 07: use normal=1, use elastic=1, use feature= ch
    - [ ] 08: use normal=1, use elastic=1, use feature= cn
------------------------------------------------
    - [ ] 09: use normal=0, use elastic=1, use feature= cdh
    - [ ] 10: use normal=0, use elastic=1, use feature= cdn
    - [ ] 11: use normal=0, use elastic=1, use feature= chn
    - [x] 12: use normal=0, use elastic=1, use feature= cdhn
    - [x] 12_1: use normal=0, use elastic=0, use feature= cdhn
------------------------------------------------
    - [ ] 13: use normal=1, use elastic=1, use feature= cdh
    - [ ] 14: use normal=1, use elastic=1, use feature= cdn
    - [ ] 15: use normal=1, use elastic=1, use feature= chn
    - [x] 16: use normal=1, use elastic=1, use feature= cdhn
    - [x] 16_1: use normal=1, use elastic=0, use feature= cdhn
- zyhseg013:在scale=50的情况下看input_feature的效果
------------------------------------------------
    - [x] 01: use normal=0, use elastic=1, use feature= c
    - [ ] 01_1: use normal=0, use elastic=0, use feature= c
    - [ ] 02: use normal=0, use elastic=1, use feature= cd
    - [ ] 03: use normal=0, use elastic=1, use feature= ch
    - [ ] 04: use normal=0, use elastic=1, use feature= cn
------------------------------------------------
    - [x] 05: use normal=1, use elastic=1, use feature= c
    - [ ] 05_1: use normal=1, use elastic=0, use feature= c
    - [ ] 06: use normal=1, use elastic=1, use feature= cd
    - [ ] 07: use normal=1, use elastic=1, use feature= ch
    - [ ] 08: use normal=1, use elastic=1, use feature= cn
------------------------------------------------
    - [ ] 09: use normal=0, use elastic=1, use feature= cdh
    - [ ] 10: use normal=0, use elastic=1, use feature= cdn
    - [ ] 11: use normal=0, use elastic=1, use feature= chn
    - [x] 12: use normal=0, use elastic=1, use feature= cdhn
    - [ ] 12_1: use normal=0, use elastic=0, use feature= cdhn
------------------------------------------------
    - [ ] 13: use normal=1, use elastic=1, use feature= cdh
    - [ ] 14: use normal=1, use elastic=1, use feature= cdn
    - [ ] 15: use normal=1, use elastic=1, use feature= chn
    - [x] 16: use normal=1, use elastic=1, use feature= cdhn
    - [ ] 16_1: use normal=1, use elastic=0, use feature= cdhn    
    
- zyhseg014:超scale的对比: scale=100 v.s. scale=50
    - [x] 01(same as 013_01) scale=50,use normal=0, use elastic=1, batch size=4
    - [x] 02 scale=80,use normal=0, use elastic=1, batch size=4
    - [x] 03 scale=150,use normal=0, use elastic=1, batch size=4
- zyhseg015: baseline totally the same
- zyhseg016: full mode(batch size 5,scale 50, 2 block reps, 32 channel)
------------------------------------------------
    - [x] 01(same as zyhseg015): use normal=0, use elastic=1, use feature= c
    - [ ] 01_1: use normal=0, use elastic=0, use feature= c
    - [ ] 02: use normal=0, use elastic=1, use feature= cd
    - [ ] 03: use normal=0, use elastic=1, use feature= ch
    - [ ] 04: use normal=0, use elastic=1, use feature= cn
------------------------------------------------
    - [x] 05: use normal=1, use elastic=1, use feature= c
    - [ ] 05_1: use normal=1, use elastic=0, use feature= c
    - [ ] 06: use normal=1, use elastic=1, use feature= cd
    - [ ] 07: use normal=1, use elastic=1, use feature= ch
    - [ ] 08: use normal=1, use elastic=1, use feature= cn
------------------------------------------------
    - [ ] 09: use normal=0, use elastic=1, use feature= cdh
    - [ ] 10: use normal=0, use elastic=1, use feature= cdn
    - [ ] 11: use normal=0, use elastic=1, use feature= chn
    - [ ] 12: use normal=0, use elastic=1, use feature= cdhn
    - [ ] 12_1: use normal=0, use elastic=0, use feature= cdhn
------------------------------------------------
    - [ ] 13: use normal=1, use elastic=1, use feature= cdh
    - [ ] 14: use normal=1, use elastic=1, use feature= cdn
    - [ ] 15: use normal=1, use elastic=1, use feature= chn
    - [ ] 16: use normal=1, use elastic=1, use feature= cdhn
    - [ ] 16_1: use normal=1, use elastic=0, use feature= cdhn    
    
    


























    
