import json, boto3
from boto3.dynamodb.conditions import Attr


dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('sagemaker-queue')


# Get a jobName -> priority mapping of all running jobs
def get_running_jobs_dict():
    # Get all running jobs
    response = table.scan(
        ProjectionExpression="jobName, priority",
        FilterExpression=Attr('jobState').eq('running')
    )
    jobs = response.get('Items', [])

    running_jobs = {job['jobName']: job['priority'] for job in jobs}

    return running_jobs

# Print the queue state
def print_queue_state():
    response = table.scan(
        ProjectionExpression='jobName, jobState, priority'
    )
    jobs = response.get('Items', [])

    print_table = []
    for job in jobs:
        print_table.append([job['jobName'], job['jobState'], job['priority']])

    # sort by priority
    sorted_table = sorted(print_table,
                         key=lambda x: -x[2])
    # Print the table
    from tabulate import tabulate
    print(tabulate(sorted_table, headers=['Job Name', 'State', 'Priority']))

# get job details
def get_job_details(job_name):
    response = table.get_item(
        Key={'jobName': job_name},
        ProjectionExpression='jobDetails'
    )
    return json.loads(response.get('Item').get('jobDetails'))

# get job state or None if the job does not exist
def get_job_state(job_name):
    response = table.get_item(
        Key={'jobName': job_name},
        ProjectionExpression='jobState'
    )
    job = response.get('Item')
    return job.get('jobState') if job else None

# update the job state
def update_job_state(job_name, new_state):
    table.update_item(
        Key={'jobName': job_name},
        UpdateExpression="SET jobState = :new_state",
        ExpressionAttributeValues={":new_state": new_state}
    )
    print(f'Update job {job_name} to {new_state}')

# remove a job entry
def remove_job(job_name):
    table.delete_item(
        Key={'jobName': job_name}
    )
    print(f'Removed job {job_name} from queue')