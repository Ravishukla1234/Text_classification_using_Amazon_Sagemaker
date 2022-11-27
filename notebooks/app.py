
import json
import os
import boto3

def lambda_handler(event, context):
    
    ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

    request_args = {}
    
    text = json.loads(json.dumps(event))

    request = {"inputs" : text["inputs"],"parameters": {"truncation": True}}
    request_args['Body'] = json.dumps(request)
    request_args['EndpointName'] = ENDPOINT_NAME
    request_args['ContentType'] = 'application/json'
    request_args['Accept'] = 'application/json'
    
    # # works successfully
    runtime= boto3.client('runtime.sagemaker')
    
    response = runtime.invoke_endpoint(**request_args)
    
    response_body = response['Body']
    
    output = json.loads(response_body.read().decode("UTF-8"))[0]
    print(output)
        # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(output)
    }
