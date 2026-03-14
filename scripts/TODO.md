# TODO

## 1 改进HO-Cap-Annotation/tools/00_convert_videos_to_h5.py

目前这个脚本有一个问题，就是加载的视频如果太大，哪怕我只选择保存特定的frame，后续的几个相机的视频也没有办法被一次性保存到h5文件里。我希望把保存到h5的这个动作变成逐个相机写入，或者使用其它的方法，不要让这个过程占用太多的存储。
目前出现的bug现象：运行./run_local.sh --sequence_name videos_0121/mallet_crush_nuts/20260121_mallet_crush_peanuts_nuts_18 --tool_name rubber_mallet --optimize 1 --object_idx 1 --hand 1 --uuid 0121
的时候只能够看到前4个相机的结果，应该是这一步没有把后面四个相机的图像正确的存储到h5文件里面HO-Cap-Annotation/data/videos_0121/mallet_crush_nuts/20260121_mallet_crush_peanuts_nuts_18/data00000000.h5


## 2 让./run_local.sh对应的这个pipeline对单个相机适配

现在的pipeline对于多个相机使用ransac来筛选pose，我希望后续的pipeline里面每一个脚本都对于只有一个相机的情况适配。