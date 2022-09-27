# 项目说明
本项目利用opencv库实现了视频的引导滤波算法，并分别使用用openmp和cuda对视频处理进行了加速。在用cuda进行加速时采用了读写线程、锁页内存等优化方式。经测试，对于同一段视频，未经优化、使用openmp和使用cuda的平均每帧处理时间为110.67ms、39.32ms、4.40ms。