import json
from utility_functions import remove_job, get_job_state
from create_table_entry import add_job_entry
from training_job_queue_manager import strip_qstamp, manage_queue


def lambda_handler(event, context):
    # identify source of event and take appropriate action
    if 'requestContext' in event and 'apiId' in event['requestContext']:
        print('Lambda triggerred by API Gateway')
        job_details = json.loads(event.get('body'))
        add_job_entry(job_details)
    elif 'source' in event and event['source'] == 'aws.sagemaker':
        print('Lambda triggerred by SageMaker job state change')
        job_name = event['detail']['TrainingJobName']
        job_status = event['detail']['TrainingJobStatus']
        print(f'{job_name} status changed to {job_status}')

        # strip qstamp from job_name
        job_name = strip_qstamp(job_name)

        if job_status in ['Completed' , 'Failed']:
            remove_job(job_name)
        elif job_status == 'Stopped':
            # check if it was manually stopped or preempted by queue manager
            if get_job_state(job_name) == 'preempted':
                print(f'job {job_name} preemption completed')
            else:
                print(f'job {job_name} {job_status}, remove from queue')
                remove_job(job_name)

    # in all cases invoke queue manager
    manage_queue()