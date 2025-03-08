import os
import ray

class RayLoggingSilencer: # HONESTLY THIS AINT DOING JACK SHIT
    def __enter__(self):
        # Backup current environment variables
        self.old_env = os.environ.copy()
        # Set the RAY_DEDUP_LOGS environment variable
        os.environ["RAY_DEDUP_LOGS"] = "0"
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Restore previous environment variables
        os.environ.clear()
        os.environ.update(self.old_env)