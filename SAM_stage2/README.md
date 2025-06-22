2 SAM生成点提示

代码：/data/segment-anything-main/segment-anything-main/point_cut.py

运行：python3 point_cut.py

3 SAM生成方框提示

代码：/data/segment-anything-main/segment-anything-main/box_cut.py

运行：python3 box_cut.py

4 SAM筛选

# stage2: Pseudo-label Generation

Click the links below to download the checkpoint here and put them in the current directory.

**[**`default` or `vit_h`**  ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

vit_l: [ ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

vit_b: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## SAM Segmentation via Point Prompt


## SAM Segmentation via Box Prompt


## Initial Reliable Pseudo-label Selection


代码：/data/segment-anything-main/segment-anything-main/similarity_test.py

运行：python3 similarity_test.py

注意打开’第二阶段融合’部分代码的注释，将’模型筛选相关代码’注释掉


Refined Pseudo-label Generation
