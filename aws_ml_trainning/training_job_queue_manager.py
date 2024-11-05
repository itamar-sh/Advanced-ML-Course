import boto3, datetime
from utility_functions import get_job_details, update_job_state, get_running_jobs_dict
from get_pending_jobs import get_pending_jobs


# set the limit on total number of instances/jobs
MAX_CAPACITY = 2

sagemaker = boto3.client('sagemaker')

# apply a queue stamp to identify that the job came from the queue
def apply_qstamp(job_name):
    return f'{job_name}-qstamp-{datetime.now().strftime("%d%H%M")}'

# strip the queue stamp
def strip_qstamp(job_name):
    return job_name.split('-qstamp-')[0]

# start a SageMaker job and update job entry in queue
def start_job(job_name):
    print(f'start job {job_name}')
    job_details = get_job_details(job_name)
    job_details['TrainingJobName'] = apply_qstamp(job_name)
    if(job_details):
        # start job with detail from queue
        # (you may optinally overwrite fields such as the iam role)
        response = sagemaker.create_training_job(**job_details)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            print(f'started job {job_name}')
            update_job_state(job_name, 'running')

# preempt a SageMaker job and update job entry in queue
def preempt_job(job_name):
    print(f'preempt job {job_name}')
    response = sagemaker.stop_training_job(TrainingJobName=job_name)
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f'preempted job {job_name}')
        update_job_state(strip_qstamp(job_name), 'preempted')

# get SageMaker jobs
def get_sagemaker_jobs(status):
    running = sagemaker.list_training_jobs(StatusEquals=status)
    return running.get('TrainingJobSummaries', [])

# queue manager
def manage_queue():
    # extract pending jobs to run
    pending = get_pending_jobs()

    if not pending:
        return

    if len(pending) > MAX_CAPACITY:
        pending = pending[:MAX_CAPACITY]

    # get running sagemaker jobs
    running = get_sagemaker_jobs('InProgress')
    total_running = len(running)

    # get stopping sagemaker jobs
    stopping = get_sagemaker_jobs('Stopping')
    total_stopping = len(stopping)

    # calculate the number of free instances
    free_slots = MAX_CAPACITY - total_running - total_stopping

    jobs_to_start = min(len(pending), free_slots)

    # for each free instance, start a job
    for i in range(jobs_to_start):
        start_job(pending[i].get('jobName'))

    still_pending = pending[jobs_to_start:]

    if not still_pending:
        return

    # assume that 'total_stopping' number of jobs will start soon
    test_for_preemption = len(still_pending) - total_stopping
    if test_for_preemption <= 0:
        return

    # check if preemption is required
    test_priority = still_pending[total_stopping:]

    running_jobs = get_running_jobs_dict()
    priority_dict = {}
    for job in running:
        job_name = job['TrainingJobName']
        priority_dict[job_name] = running_jobs[strip_qstamp(job_name)]

    # sort running jobs from lowest to highest priority
    sorted_running = sorted(priority_dict.items(), key=lambda item: item[1])

    index = 0
    while index < test_for_preemption and \
          test_priority[index].get('priority') > sorted_running[index][1]:
        preempt_job(sorted_running[index][0])
        index = index + 1
        