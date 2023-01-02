FROM python:3.8

WORKDIR /tensorslow

COPY . .

RUN pip install -r requirements.txt

CMD tail -f /dev/null