FROM python:3.12-slim AS encoder-build
WORKDIR /build
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt requirements-public.txt requirements-hosted.txt requirements-export.txt ./
RUN pip install --no-cache-dir -r requirements-export.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
COPY foodvision ./foodvision
COPY models/demo_model.* ./models/
COPY examples ./examples
COPY scripts/export_hosted_encoder.py ./scripts/export_hosted_encoder.py
RUN python scripts/export_hosted_encoder.py --output /encoder

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PORT=8080 AFTERMEAL_HOSTED_UPLOADS=1
COPY requirements.txt requirements-public.txt requirements-hosted.txt ./
RUN pip install --no-cache-dir -r requirements-hosted.txt
COPY foodvision ./foodvision
COPY web ./web
COPY data ./data
COPY models ./models
COPY --from=encoder-build /encoder ./models/hosted
COPY examples ./examples
COPY reports/benchmark.json ./reports/benchmark.json
COPY gunicorn.conf.py LICENSE ./
COPY docs/DATA.md ./docs/DATA.md
RUN useradd --uid 10001 --create-home aftermeal
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/healthz',timeout=2)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "foodvision.public:create_app()"]
