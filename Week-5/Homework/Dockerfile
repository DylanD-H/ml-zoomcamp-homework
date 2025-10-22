FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev
ENV PATH="/code/.venv/bin:$PATH"

COPY ["predict.py", "./"]

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]