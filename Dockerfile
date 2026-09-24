FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 PORT=8080
COPY requirements.txt requirements-public.txt ./
RUN pip install --no-cache-dir -r requirements-public.txt
COPY foodvision ./foodvision
COPY web ./web
COPY data ./data
COPY models ./models
COPY examples ./examples
COPY reports/benchmark.json ./reports/benchmark.json
COPY gunicorn.conf.py LICENSE ./
COPY docs/DATA.md ./docs/DATA.md
RUN useradd --uid 10001 --create-home aftermeal
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/healthz',timeout=2)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "foodvision.public:create_app()"]
