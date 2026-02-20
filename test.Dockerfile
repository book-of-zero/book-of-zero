ARG MY_VAR="hello_world"

FROM alpine:latest
RUN echo "Value without re-declaring: '${MY_VAR}'"

FROM alpine:latest
ARG MY_VAR
RUN echo "Value with re-declaring: '${MY_VAR}'"
