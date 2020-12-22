### Build
```
sudo docker build -t antidictapi .
```

### Run
```
sudo docker run -v "$(pwd)"/models:/app/models -p 5001:5000 antidictapi
```

### Test

```
curl -i -X POST \
-H "Content-Type: application/json; indent=4" \
-d '{
    "jsonrpc": "2.0",
    "method": "process",
    "params": {"text": "Пример большого текста для примера!"},
    "id": "1"
}' http://localhost:5001/
```