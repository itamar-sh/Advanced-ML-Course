import boto3
from boto3.dynamodb.conditions import Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('sagemaker-queue')

# Get a list of all pending jobs sorted by priority
def get_pending_jobs():
    response = table.scan(
        ProjectionExpression='jobName, priority, entryTime',
        FilterExpression=Attr('jobState').ne('running')
    )
    jobs = response.get('Items', [])

    # sort jobs, first by priority (descending) and then by entryTime
    sorted_jobs = sorted(jobs,
                         key=lambda x: (-x['priority'], x['entryTime']))

    return sorted_jobs

