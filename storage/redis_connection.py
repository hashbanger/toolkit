import json
import redis


class RedisDB:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def save_data(self, obj, data):
        try:
            self.redis_client.set(str(obj), json.dumps(data))
            print("Saved data successfully!")
        except Exception as e:
            print("Error in saving data", e)
            raise

    def get_data(self, obj):
        try:
            data = self.redis_client.get(obj)
            if data is not None:
                data = json.loads(data)
            return data
        except Exception as e:
            print(f"Error in retrieving data: {e}")
            raise

    def flush_data(self):
        try:
            self.redis_client.flushall()
            print("Flushed all data successfully!")
        except Exception as e:
            print(f"Error in flushing data: {e}")
            raise
