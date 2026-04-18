# MCDM

MCDM: A multimodal code clone detection framework that jointly leverages source code semantics and binary-level representations. The model integrates UniXcoder and ViT with a cross-modal attention fusion mechanism to capture complementary high-level and low-level program features, improving robustness on challenging clone detection tasks.

## Usage

### 1. Environment Setup
We provide an `environment.yml` file for easy reproduction. Please create the environment using:

```bash
conda env create -f environment.yml
conda activate mcdm
```

### 2. Training & Evaluation
You can run the model using command-line arguments:

```bash
python run.py \
    --output_dir=your_output_path \
    --vit_model_name_or_path=google/vit-base-patch16-224 \
    --vit_unfrozen_layer=3 \
    --unixcoder_model_name_or_path=microsoft/unixcoder-base \
    --unixcoder_unfrozen_layer=3 \
    --do_train \
    --train_data_file=train_pairs.txt \
    --eval_data_file=test_pairs.txt \
    --test_data_file=test_pairs.txt \
    --code_file_path=SourceCode \
    --image_file_path=img \
    --epoch=10 \
    --learning_rate=1e-4 \
    --seed=42 \
    --evaluate_during_training
```

Alternatively, you can directly specify parameters inside run.py by modifying:

```bash
sys.argv = [
    "run.py",
    "--output_dir=your_output_path",
    "--vit_model_name_or_path=google/vit-base-patch16-224",
    "--vit_unfrozen_layer=3",
    "--unixcoder_model_name_or_path=microsoft/unixcoder-base",
    "--unixcoder_unfrozen_layer=3",
    "--do_train",
    "--train_data_file=train_pairs.txt",
    "--eval_data_file=test_pairs.txt",
    "--test_data_file=test_pairs.txt",
    "--code_file_path=SourceCode",
    "--image_file_path=img",
    "--epoch=10",
    "--learning_rate=1e-4",
    "--seed=42",
    "--evaluate_during_training",
]
```

### 3. Key Arguments

--output_dir: Path to save checkpoints and results
--train_data_file: Training dataset (program pairs)
--eval_data_file: Validation dataset
--test_data_file: Test dataset
--code_file_path: Directory of source code files
--image_file_path: Directory of binary image representations
--epoch: Number of training epochs
--learning_rate: Learning rate
--seed: Random seed for reproducibility

### 4. Notes

Ensure that source code files and corresponding binary images are properly aligned.
