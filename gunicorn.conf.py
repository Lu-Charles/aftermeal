"""Bounded sample-only deployment; TLS and traffic limits belong at the host proxy."""
import os
bind = '0.0.0.0:' + str(int(os.environ.get('PORT', '8080')))
workers = 2
worker_class = 'sync'
timeout = 15
graceful_timeout = 15
backlog = 64
limit_request_line = 2048
limit_request_fields = 30
limit_request_field_size = 4096
max_requests = 2000
max_requests_jitter = 100
control_socket_disable = True
accesslog = '-'
access_log_format = '%(m)s %(U)s %(s)s %(L)s'
errorlog = '-'
