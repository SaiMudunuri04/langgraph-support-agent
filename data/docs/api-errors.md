# API errors and status

A 500 error means our API is having issues on our side. Check the status page
for ongoing incidents before retrying. For 429 rate-limit errors, back off
exponentially and retry. Include the x-request-id response header when
opening a ticket so we can trace the failing call.
