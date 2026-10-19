FROM python:3.12-slim
RUN pip install --no-cache-dir "mcp>=1.9"
COPY qff_mcp.py comparison.py /app/
ENV QFF_DATA=https://delirious6423.github.io/quant-fidelity-frontier/
EXPOSE 7860
CMD ["python3", "/app/qff_mcp.py", "--http", "7860"]
