# Neural-Network

## For optimal performance, compile the project using the following cmake command:
```
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release -DUSE_THREADING=ON 
```
## Benchmark - Highest average recorded
- Intel Core i5 9th Gen: ~150,000 Data/s
- Intel Core Ultra 5: ~250,000 Data/s
