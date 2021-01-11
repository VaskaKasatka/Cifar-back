FROM python:3.7.9

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY . /usr/src/app/
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python","run.py"]