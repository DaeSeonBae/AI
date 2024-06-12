FROM python:3.12.2

# 필수 패키지 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 현재 디렉토리의 모든 파일을 /app 디렉토리에 복사
COPY . /app

# requirements.txt 파일을 사용하여 필요한 pip 패키지 설치
RUN pip install -r /app/requirements.txt

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 8000번 포트를 외부에 노출
EXPOSE 8000

# 컨테이너 시작 시 실행할 명령어 설정 (manage.py runserver 사용)
ENTRYPOINT ["python", "manage.py", "runserver","0.0.0.0:8000"]