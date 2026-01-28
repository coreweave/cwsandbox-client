"""Compatibility patches for third-party libraries.

This module patches ConnectRPC's JSON codec to ignore unknown protobuf fields.
The backend may return fields not yet in the client's proto definitions,
and by default protobuf's JSON parser raises ParseError on unknown fields.

This patch must be applied before any ConnectRPC clients are created.
"""

from __future__ import annotations

from typing import TypeVar

import connectrpc._codec as codec_module
from google.protobuf.json_format import Parse as MessageFromJson
from google.protobuf.message import Message

V = TypeVar("V", bound=Message)


class _LenientProtoJSONCodec(codec_module.ProtoJSONCodec[V]):
    """ProtoJSONCodec that ignores unknown fields in JSON responses.

    The backend may add new fields to protobuf messages before the client's
    proto definitions are updated. Without this patch, the client would fail
    with ParseError when receiving responses with unknown fields.
    """

    def decode(self, data: bytes | bytearray, message: V) -> V:
        MessageFromJson(bytes(data), message, ignore_unknown_fields=True)
        return message


def _patch_connectrpc_codec() -> None:
    """Replace ConnectRPC's global JSON codec with our lenient version."""
    lenient_codec: _LenientProtoJSONCodec[Message] = _LenientProtoJSONCodec()

    codec_module._proto_json_codec = lenient_codec
    codec_module._codecs[codec_module.CODEC_NAME_JSON] = lenient_codec
    codec_module._codecs[codec_module.CODEC_NAME_JSON_CHARSET_UTF8] = lenient_codec


_patch_connectrpc_codec()
