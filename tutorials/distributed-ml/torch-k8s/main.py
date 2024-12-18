import os
import json
import time

if __name__ == "__main__":
    print(json.dumps(dict(os.environ), indent=4))
    # This ensures that the Master does not die before the Worker starts up,
    # since the Worker will look for the master. 
    time.sleep(3600)
    print("FINISHED")
    # print("RANK: " + os.environ["RANK"])
    # print("LOCAL_RANK: " + os.environ["LOCAL_RANK"])
    # print("MASTER_PORT: " + os.environ["MASTER_PORT"])
    # print("MASTER_ADDR: " + os.environ["MASTER_ADDR"])
    # print("WORLD_SIZE: " + os.environ["WORLD_SIZE"])