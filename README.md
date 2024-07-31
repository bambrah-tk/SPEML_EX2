
Just example you might need to try around
# Step 1: Setup the environment
```
pip install -r requirements.txt
```

# Step 2: Clone the repository (if not already done)


# Step 3: Prepare the trigger set
```
python prepare_trigger_set.py
```

# Step 4: Train the models
Training model without watermark:
```
python train.py --dataset cifar10 --max_epochs 60
```

Training model with watermark:
```
python train.py --wmtrain -wmt --dataset cifar10 --max_epochs 60
```

# Step 5: Evaluate the models
Evaluating original model accuracy:
```
python evaluate.py --model_path checkpoint/original_model.t7 --dataset cifar10 --test_db_path ./data
```

Evaluating watermarked model accuracy:
```
python evaluate.py --model_path checkpoint/watermarked_model.t7 --dataset cifar10 --test_db_path ./data
```

Evaluating watermark effectiveness
```
python evaluate.py --model_path checkpoint/watermarked_model.t7 --wm_path ./data/trigger_set/ --wm_lbl labels-cifar.txt
```

# Step 6: Fine-tune the model and evaluate robustness
Fine-tuning the watermarked model:
```
python fine_tune.py --model_path checkpoint/watermarked_model.t7 --dataset cifar10 --train_db_path ./data --test_db_path ./data --epochs 5
```

Evaluating fine-tuned model for watermark effectiveness:
```
python evaluate.py --model_path checkpoint/fine_tuned_model.t7 --wm_path ./data/trigger_set/ --wm_lbl labels-cifar.txt
```

