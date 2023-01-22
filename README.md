## JCG

Implementation of JCG, A Joint Framework of Structural and Semantic Contrastive Learning for Graph Collaborative Filtering


### Main File Contents

- `/graph_Augmentation` implements edge-perturbation-based graph augmentation.
- `/reckit` contains CPP-based auxiliary packages for graph augmentation.
- `/properties` contains model training settings for various datasets.
- `JCG.py` is the framework that integrates a GCF backbone with joint CL strategies
- `trainer.py` implements the training and validating progresses for JCG.
- `main.py` load dataset and training settings for JCG.

### How to use

1. Install required packages (Using a virtual environment is recommended).
   ````python
   pip install -r requirements.txt
   ````
2. Download datasets package from [here](https://pan.baidu.com/s/1SQsafQh8J3qT3LtmQz-C-A?pwd=1223), 
   then put them into the root of the project repository.
3. Run `main.py` to train JCG and obtain results in logs.
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
