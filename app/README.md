# Lung Cancer Detection Application

This Modal application contains a serverless FastAPI application with a `/predict` endpoint
for tumor detection in images of lung CT/PET scans. Below are instructions on how to deploy the application 
to Modal.

The application can be accessed at:

```bash
https://lung-cancer-detector--lung-cancer-detector-fastapi-app.modal.run/docs
```

Note that if the application is turned off, visiting the above webpage will lead to a the message:

```
modal-http: app for invoked web endpoint is stopped
```

Starting the appication requires manual deployment from a local machine. Contact Raamis Hussain to deploy the application or follow the instructions below to set up Modal and deploy your own app.

## Modal Setup

If you want to deploy your own application, you must first have a modal account and follow the set up to get your local machine ready for deployment. Follow the instructions here to get set up: [Modal Introduction](https://modal.com/docs/guide)


## Uploading Model Weights

Once you have set up your account, you will need to upload model weights to a persistent volume. This is relatively cheap, according to [Modal pricing](https://modal.com/pricing), the price for storage is `$0.00000222 / GiB / sec`. The model weights for our YOLOv5 model are only 14MB, which means storing these weights in persistent storage will only cost roughly `$0.0025/day`.

To upload model weights, use the `upload_weights.py` script in `/app`. Provide arguments for volume name, the local path to the model weights, and the desired path to the weights on the remote volume. An example call may look like:

```bash
python -m app.upload_weights --local ./model/weights/best.pt --remote /best.pt --volume lung-cancer-detector
```

Note that `REMOTE_WEIGHT_PATH` in `config.py` must match the location of the stored weights on the remote volume. The current setup of the application mounts the persistent volume to `/model` which means if you upload weights to `/best.pt` as shown above, `REMOTE_WEIGHT_PATH` must be `/model/best.pt`, otherwise they weights will not be found and the model will fail to load. 

## API Key

The application requires an API key when trying to run inference via the `/predict` endpoint. If you are deploying your own application, you will need to create an API key and store it in a `.env` file at the root level of this repo. You must also create a Modal secret which will set the API key during deployment. Your `.env` file should have the following contents.

```bash
LUNG_CANCER_API_KEY=<YOUR_API_KEY>
```

Then you can create the Modal secret by running the following command:

```bash
modal secret create --from-dotenv .env lung_cancer_api_key
```

This will push the secret to your current Modal workspace and it will be available as an environment variable after you application is deployed. 

## Deployment

Once you have followed the above steps, you should be ready to build and deploy the application. Simply run the following command and the application will build and create a web function with a URL which you can visit to access the swagger docs of the FastAPI app.

```bash
modal deploy -m app.app
```

Note: make sure to append `/docs` to the end of the generated URL to access the swagger docs. Then you can authenticate with your API key and start uploading images to the `/predict` endpoint to run inference. 
