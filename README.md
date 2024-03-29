# PA-LSTM-relation-extraction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/position-aware-attention-and-supervised-data/relation-extraction-on-tacred)](https://paperswithcode.com/sota/relation-extraction-on-tacred?p=position-aware-attention-and-supervised-data)

Implementation of [Position-aware Attention and Supervised Data Improve Slot Filling](https://www.aclweb.org/anthology/D17-1004.pdf).

## Environment Requirements
* python 3.6
* pytorch 1.3.0

## Data
* [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24)
* [glove.6B.200d.txt](https://nlp.stanford.edu/projects/glove/)

## Usage
1. Purchase the dataset and move it in the `data` folder.
2. Download the embedding and move it in the `embedding` folder. The download script is available in the `embedding` folder.
3. Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

## Result
The result of my version and that in paper are present as follows:
| paper | my version |
| :------: | :------: |
| 0.651 | 0.6503 |

The training log can be seen in `train.log`.

*Note*:
* Some settings are different from those mentioned in the paper.
* If you cannot achieve the same result as mine, please check the version of Pytorch.
* Some errors exists in my code, but on the whole it is right.

## Reference Link
* https://github.com/yuhaozhang/tacred-relation
