FROM public.ecr.aws/lambda/python:3.13

RUN pip install pipenv

WORKDIR ${LAMBDA_TASK_ROOT}

COPY Pipfile Pipfile.lock ./

RUN pipenv requirements > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY src/ ./

CMD ["predict.lambda_handler"]