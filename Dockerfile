FROM vllm/vllm-openai:v0.21.0 AS wheelhouse

RUN mkdir -p /wheelhouse && \
    python3 -m pip download \
        "mineru[core]>=3.4.0" \
        --dest /wheelhouse \
        --prefer-binary \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --timeout 300 \
        --retries 20


FROM vllm/vllm-openai:v0.21.0 AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=wheelhouse /wheelhouse /opt/wheels

RUN python3 -m pip install \
        --no-index \
        --find-links=/opt/wheels \
        "mineru[core]>=3.4.0" \
        --break-system-packages && \
    rm -rf /opt/wheels

ENV MINERU_MODEL_SOURCE=local

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]