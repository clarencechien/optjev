#!/usr/bin/env bash
# Wrapper: map this environment's token env vars to Modal's official names.
export MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-$modal}"
export MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-$modal_secret}"
exec modal "$@"
