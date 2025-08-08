import argparse
import os

from modal import Volume


def upload_weights(local_path: str, remote_path: str, volume_name: str):
    """Upload local model weights to remote Modal volume
    
    Args:
        local_path: str
            local path to model weights
        remote_filename: str
            Name of weights file to be save in model/ on volume
        volume_name: str
            Name of modal volume 
    """

    volume = Volume.from_name(volume_name, create_if_missing=True)

    if not os.path.isfile(local_path):
        print(f"File not found at {local_path}")
        return


    with volume.batch_upload() as batch:
        batch.put_file(local_path,remote_path)
    print(f"Successfully uploaded weights to {remote_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local",
        type=str,
        help="Local path to model weights",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="path on remote volume to upload to"
    )
    parser.add_argument(
        "--volume",
        type=str,
        help="Name of modal volume"
    )

    args = parser.parse_args()

    upload_weights(args.local,args.remote,args.volume)

