## TO RUN SERVER USE THE COMMAND
```
uvicorn api.app:app --reload
```
## USE THE FastAPI docs TO SEND REQUESTS
```
http://127.0.0.1:8000/docs
```
### USE THE FOLLOWING FORMAT
```
{
  "traffic": 200,
  "latency": 50
}
```
- OR use this through curl 
```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"traffic": 200, "latency": 50}'
```
## AND THE UNIT TESTING HAS BEEN DONE ALREADY 
```
python -m unittest discover -s tests -p "test_*.py"

```


## AUTHORS
- [Tejanshu Bhandari](https://github.com/TLxGHOST)
- [Prashant Raj](https://github.com/prashantraj1X1)
- [Gaurav Kumar](https://github.com/Gaurav-kr7103)
- [Harsh Ray](https://github.com/HarshRay04)
