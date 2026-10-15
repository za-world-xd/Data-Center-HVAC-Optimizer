FROM python:3.13-slim
WORKDIR /opt/cooltwin
COPY pyproject.toml requirements.txt ./
COPY LICENSE ./
COPY engine ./engine
COPY app ./app
COPY examples ./examples
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir . && useradd --create-home cooltwin && mkdir -p data && chown -R cooltwin /opt/cooltwin
USER cooltwin
EXPOSE 8501 8000
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]
