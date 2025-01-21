## Official Implementation for NDSS'2025 paper *PBP: Post-training Backdoor Purification for Malware Classification*
DOI: 10.14722/ndss.2025.240603

In recent years, the rise of machine learning (ML) in cybersecurity has brought new challenges, including the increasing threat of backdoor poisoning attacks on ML malware classifiers. These attacks aim to manipulate model behavior when provided with a particular input trigger. For instance, adversaries could inject malicious samples into public malware repositories, contaminating the training data and potentially misclassifying malware by the ML model. Current countermeasures predominantly focus on detecting poisoned samples by leveraging disagreements within the outputs of a diverse set of ensemble models on training data points. However, these methods are not suitable for scenarios where Machine Learning- as-a-Service (MLaaS) is used or when users aim to remove backdoors from a model after it has been trained. Addressing this scenario, we introduce PBP, a post-training defense for malware classifiers that mitigates various types of backdoor embeddings without assuming any specific backdoor embedding mechanism. Our method exploits the influence of backdoor attacks on the activation distribution of neural networks, independent of the trigger-embedding method. In the presence of a backdoor attack, the activation distribution of each layer is distorted into a mixture of distributions. By regulating the statistics of the batch normalization layers, we can guide a backdoored model to perform similarly to a clean one. Our method demonstrates substantial advantages over several state-of-the-art methods, as evidenced by experiments on two datasets, two types of backdoor methods, and various attack configurations. Our experiments showcase that PBP can mitigate even the SOTA backdoor attacks for malware classifiers, e.g., Jigsaw Puzzle, which was previously demonstrated to be stealthy against existing backdoor defenses. 
![PBP-Pipeline](pbp-overall.png)

### Dataset Preparation
**EMBER-v1 dataset**
- Following instructions from the GitHub repo for EMBER dataset, we can download and extract EMBER dataset following the instruction of the original paper at [Explanation-guided Backdoor Attacks](https://github.com/ClonedOne/MalwareBackdoors).

- After you can achieve the extracted EMBER data, put the four files `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat` into `datasets/ember` folder as follows.
```
datasets/
└── ember/
    └── np
        ├── watermarked_X_test_32_feats.npy
        └── wm_config_32_feats.npy
    ├── y_train.dat
    ├── X_train.dat
    ├── y_train.dat
    ├── X_test.dat
    └── y_test.dat
```
- For easy reproducibility objective, we provide the processed data EMBER-v1 [here](https://zenodo.org/records/14253945?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlmNjRiOWFkLTIyMWUtNGVmZi04NmJmLTBhODg4YTkxMDJkZiIsImRhdGEiOnt9LCJyYW5kb20iOiI3YzQ5NzhmZTY5YjA1OTZjMjY3NWM2ZmI0ODYyN2U5YiJ9.a4WZjxQa0MdCvLkcgWmN18HM0GDJKYf4mrdiYO-mzz0xxbsu-J6B2uEhNcNyvpEY85Xth1rKEIk85f5ugfvKhA).

**AndroZoo dataset**
- We encourage interested readers to refer to the original authors of JIGSAW backdoor attacks to acquire preprocessed data for AndroZoo at [Explanation-guided Backdoor Attacks](https://github.com/ClonedOne/MalwareBackdoors). Our code uses the data shared by these authors, so we are not able to distribute it publicly.
- After you acquire the pre-processed datasets for AndroZoo, put it in folder `datasets/apg/` as follows:
```
datasets/
└── apg/
    ├── apg_sha_family.csv
    ├── apg-meta.json
    ├── apg-X.json
    ├── apg-y.json
    └── family_cnt.csv
```

### Requirements
Here are the packages required to reproduce our results
```
>>> torch.__version__
'2.4.1'
Python 3.10.12
```
For other main packages please refer to `requirements.txt`.
To easily setup for the conda environment,
```
conda config --set channel_priority flexible &&
conda env create --file=environment.yml &&
conda activate pbp-code
```

### Checkpoints
Download the clean version of the EMBER dataset [here](https://zenodo.org/records/14253945?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlmNjRiOWFkLTIyMWUtNGVmZi04NmJmLTBhODg4YTkxMDJkZiIsImRhdGEiOnt9LCJyYW5kb20iOiI3YzQ5NzhmZTY5YjA1OTZjMjY3NWM2ZmI0ODYyN2U5YiJ9.a4WZjxQa0MdCvLkcgWmN18HM0GDJKYf4mrdiYO-mzz0xxbsu-J6B2uEhNcNyvpEY85Xth1rKEIk85f5ugfvKhA). 

Then put it at [models/ember/torch/embernn](models/ember/torch/embernn).


### [E1] Training and Fine-tuning with EMBER dataset
- First, train a backdoor model by running `./train_backdoor_ember.sh`. This will generate four backdoored models corresponding to different poisoning ratios. The models should be saved at [saved_model_path](models/ember/torch/embernn/backdoor).

```
------- Final Evaluation -------
+-------------------+---------+
| Metric            |   Value |
+===================+=========+
| Main Accuracy     | 99.0135 |
+-------------------+---------+
| Backdoor Accuracy | 95.067  |
+-------------------+---------+
| Poisoning Rate    |  0.01   |
+-------------------+---------+
```

- Second, to fine-tune these models with different fine-tuning methods: run `./experiment1_finetune_backdoor_ember.sh`. This will generate six fine-tuned models corresponding to each poisoning ratio. The models should be saved at [saved_ft_model_path](models/ember/torch/embernn/).

The successful results are presented in the following table:
```
------- Fine-tuning Evaluation -------
+-----------+------------------+------------------------+
| Mode      |   Clean Accuracy |   Adversarial Accuracy |
+===========+==================+========================+
| ft        |          99.003  |                96.2041 |
+-----------+------------------+------------------------+
| ft-init   |          99.092  |                96.9629 |
+-----------+------------------+------------------------+
| fe-tuning |          99.063  |                97.2255 |
+-----------+------------------+------------------------+
| lp        |          99.0605 |                92.2477 |
+-----------+------------------+------------------------+
| fst       |          98.9795 |                96.4668 |
+-----------+------------------+------------------------+
| proposal  |          96.8865 |                11.4564 |
+-----------+------------------+------------------------+
Completed in: 0:28:31.483780 seconds.
------- ********************** -------
```

### [E2] Training and Fine-tuning with AndroZoo dataset
- First, train a backdoor model by running `./train_backdoor_jigsaw.sh`. This will generate four backdoored models corresponding to different poisoning ratios. The models should be saved at [saved_model_path](models/ember/torch/embernn/backdoor).
- Second, to fine-tune these models with different fine-tuning methods: run `./experiment2_finetune_backdoor_jigsaw.sh`. This will generate six fine-tuned models corresponding to each poisoning ratio. The models should be saved at [saved_ft_model_path](models/ember/torch/embernn/).


### Main Hyper-parameters Table
| Name                | Type      | Description                                                                                     | Default                          |
|---------------------|-----------|-------------------------------------------------------------------------------------------------|----------------------------------|
| `device`            | `str`     | The device to train on (e.g. `'cpu'`, `'cuda:0'`).                                              | `"cuda"`                         |
| `ft_mode`           | `str`     | Fine-tuning mode (`'all'`, `'fe-tuning'`, etc.).                                                | `"fe-tuning"`                    |
| `num_classes`       | `int`     | Number of classes in the dataset.                                                                | `2`                             |
| `epochs`            | `int`     | Number of training epochs.                                                                       | `10`                             |
| `dataset`           | `str`     | Name of the dataset to be used.                                                                  | `"malimg"`                       |
| `dataset_path`      | `str`     | Path to the dataset directory.                                                                   | `"datasets/malimg"`     |
| `folder_path`       | `str`     | Path to the folder where models are saved.                                                       | `"models/malimg/torch"` |
| `attack_target`     | `int`     | The target class for backdoor attacks.                                                           | `0`          |
| `batch_size`        | `int`     | Batch size for training.                                                                         | `64`                             |
| `test_batch_size`   | `int`     | Batch size for testing.                                                                          | `512`                            |
| `random_seed`       | `int`     | Random seed for reproducibility.                                                                 | `12`         |
| `model`             | `str`     | Model architecture to use.                                                                       | `"embernn"`                  |
| `log`               | `bool`    | Whether to log outputs to a file.                                                                | `False`     |
| `initlr`            | `float`   | Initial learning rate if using a learning rate scheduler.                                        | `None`      |                                                            | `False`     |
| `save`              | `bool`    | Whether to save the trained model.                                                               | `False`     |
| `linear_name`       | `str`     | Name of the linear layer (last layer), if different from the default.                                         | `"classifier"`                   |
| `alpha`             | `float`   | Hyperparameter to balance loss terms (e.g. in feature shift regularization).                    | `0.2`       |
| `f_epochs`          | `int`     | Number of fine-tuning epochs.                                                                    | `5`                              |
| `f_lr`              | `float`   | Fine-tuning learning rate.                                                                       | `0.0002`                         |
| `target_label`      | `int`     | Target label used in attacks.                                                                    | `0`                              |
| `is_backdoor`       | `int`     | Indicator whether the dataset contains backdoored samples (1 for yes, 0 for no).                 | `1`                              |
| `lr`                | `float`   | Learning rate for the optimizer.                                                                 | `0.002`                          |
| `ft_size`           | `float`   | Portion of the dataset to use for fine-tuning.                                                   | `0.05`                           |
| `poison_rate`        | `float`     | Poisoning rate for training samples.                                                                      | `0.01`                              |

*You can modify the training and fine-tuning configuration. The configs can be found in [this](configs/backdoors)*

### Acknowledgement
[Jigsaw Puzzle Backdoor Attacks](https://github.com/whyisyoung/JigsawPuzzle) 

[Explanation-guided Backdoor Attacks](https://github.com/ClonedOne/MalwareBackdoors)
