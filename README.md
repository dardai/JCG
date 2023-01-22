## JCG

Implementation of JCG, A Joint Framework of Structural and Semantic Contrastive Learning for Graph Collaborative Filtering


### Main File Contents

- `/bug_Augmentation/preprocess.py` includes text cleaning, tokenization and parts for bug report preprocessing.
- `/bug_Augmentation/augment_main.py` implement prototype clustering-based augmentation.
- `pcg.py` is the joint model of GCF backbone and semantic CL module
- `main.py` is the training and validating progresses for PCG.

### How to use

1. Install required packages (Using a virtual environment is recommended).
   `pip install -r requirements.txt`
2. Download datasets package from [here](https://pan.baidu.com/s/1SQsafQh8J3qT3LtmQz-C-A?pwd=1223), 
   then put them into the root of the project repository.
3. Run `main.py`.
   ```python
   python main.py --dataset jester
   ```

### Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following papers as the references if you use our codes or the processed datasets.

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```
