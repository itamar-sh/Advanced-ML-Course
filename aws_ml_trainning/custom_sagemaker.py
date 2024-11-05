from sagemaker.pytorch import PyTorch
from sagemaker.session import Session
import requests, logging
logger = logging.getLogger('sagemaker')

def submit_to_training_queue(job):
    logger.info(f"Adding training-job {job['TrainingJobName']} to queue")
    logger.debug('train request: {json.dumps(job, indent=4)}')

    region = 'us-west-2'
    api_id = '<api-id>'  # Set this to your API Gateway ID
    url = f'https://{api_id}.execute-api.{region}.amazonaws.com/prod/add-job'
    headers = {'x-apigw-api-id': '<api-id>'} # insert api gateway id

    # submit job
    response = requests.post(url, headers=headers, json=job)

class QueueTrainingJobSession(Session):
    def _intercept_create_request(self, request, create, func_name = None):
        """This function intercepts the create job request

        Args:
          request (dict): the create job request
          create (functor): a functor calls the sagemaker client create method
          func_name (str): the name of the function needed intercepting
        """
        if func_name == 'train':
            submit_to_training_queue(request)
        else:
            super()._intercept_create_request(request,create,func_name)

# define job
estimator = PyTorch(
    role='<sagemaker role>',
    entry_point='train.py',
    instance_type='ml.p5.48xlarge',
    instance_count=1,
    framework_version='2.0.1',
    py_version='py310',
    tags=[{'Key': 'priority', 'Value': '100'}],
    keep_alive_period_in_seconds=60, # keep warm for 1 minute
    # use our custom Session class
    sagemaker_session=QueueTrainingJobSession()
)

estimator.fit(wait=False)
