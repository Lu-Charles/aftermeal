"""One model process with spare threads for health and sample requests."""
import os
bind = '0.0.0.0:' + str(int(os.environ.get('PORT', '8080')))
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 180
graceful_timeout = 15
backlog = 64
limit_request_line = 2048
limit_request_fields = 30
limit_request_field_size = 4096
max_requests = 0
max_requests_jitter = 0
control_socket_disable = True
accesslog = '-'
access_log_format = '%(m)s %(U)s %(s)s %(L)s'
errorlog = '-'
