# Multiple-Crowd-abnormal-behavior-detection
Deep learning-based multi-label crowd abnormal behavior detection, detection range includes crowd gathering (prone to trampling events), destruction of buildings, religious assembly speeches, arson, etc.
基于深度学习的多标签人群异常行为检测，检测范围包括人群聚集(易发生踩踏事件)，破坏建筑物，宗教集会演讲，纵火等。
本项目需要人工手动对数据进行标注，数据机为视频形式，将标注结果存为xlsx格式，第一列为标签出现的时间范围（例如00:00-01:32），后面几列为该时间范围内出现的标签（人群聚集，政府楼，破坏等）。
1. 对标注处理的代码为 process_img.py，可以将xlsx中的标注和输入视频帧一一对应起来，并生成标注的csv格式文件。
2. 训练代码为 notebook_model.py，在其中可以更改训练的batch size，epoch数以及其他参数。训练结束后会生成pth模型文件。
3. 测试以及验证代码为 test.py，本代码中validation选取的事全是数据中的部分，先生成test的img结果，再把img结果合并成输出的视频结果，如果需要其他数据进行测试可以自行更改相应路径。
