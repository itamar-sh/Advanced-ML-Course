import json, boto3, datetime

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('sagemaker-queue')


# Called by add-job REST API method
def add_job_entry(job_json):
    """
    We assume that request contains the same contents as the input to the create_training_job API in JSON format.
    We further assume that the priority of the workload is entered as a key-value tag in the training job definition.
    """
    job_details = json.loads(job_json)

    # extract job_name
    job_name = job_details['TrainingJobName']
    print(f'add entry {job_name}')

    # get current time
    entry_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # default priority is 0
    priority = 0

    # update priority based on tags
    tags = job_details['Tags']
    for tag in tags:
        if tag['Key'] == 'priority':
            priority = int(tag['Value'])
            break

    # create entry
    entry = {
       'jobName': job_name,
       'entryTime': entry_time,
       'jobState': 'pending',
       'priority': priority,
       'jobDetails': job_json
    }
    table.put_item(Item=entry) #TODO handle errors
    print(f'Added job {job_name} to queue')