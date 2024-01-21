"""
use subprocess to check language server stdio.
references:
https://github.com/windmill-labs/windmill/blob/v1.101.1/lsp/pyls_launcher.py
https://github.com/python-lsp/python-lsp-jsonrpc/blob/v1.0.0/pylsp_jsonrpc/streams.py
"""

import asyncio
import json

import logging
import multiprocessing
import subprocess
import sys
import threading
import os
from typing import List, Optional

import aiohttp
from aiohttp import web
import ssl
from contextlib import suppress


async def cancel_task(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class AsyncJsonRpcStreamReader:

    def __init__(self, reader: asyncio.StreamReader):
        self._rfile = reader

    # def close(self):
    #     self._rfile.close()

    async def listen(self, message_consumer):
        """Blocking call to listen for messages on the rfile.

        Args:
            message_consumer (fn): function that is passed each message as it is read off the socket.
        """
        async for line in self._rfile:
            content_length = self._content_length(line)
            while line and line.strip():
                line = await self._rfile.readline()
            if line == b"" or content_length is None:
                break
            request_str = await self._rfile.readexactly(content_length)
            try:
                await message_consumer(json.loads(request_str.decode('utf-8')))
            except ValueError:
                log.exception("Failed to parse JSON message %s", request_str)
                continue

    @staticmethod
    def _content_length(line):
        """Extract the content length from an input line."""
        if line.startswith(b'Content-Length: '):
            _, value = line.split(b'Content-Length: ')
            value = value.strip()
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(
                    "Invalid Content-Length header: {}".format(value)) from e

        return None


import rich


class AsyncJsonRpcStreamWriter:

    def __init__(self, wfile: asyncio.StreamWriter, **json_dumps_args):
        self._wfile = wfile
        self._wfile_lock = asyncio.Lock()
        self._json_dumps_args = json_dumps_args

    async def close(self):
        async with self._wfile_lock:
            self._wfile.close()

    async def write(self, message):
        async with self._wfile_lock:
            if self._wfile.is_closing():
                return
            try:
                # print("JSONRPC OUT", message)
                body = json.dumps(message, **self._json_dumps_args)

                # Ensure we get the byte length, not the character length
                content_length = len(body) if isinstance(body, bytes) else len(
                    body.encode('utf-8'))
                response = (
                    "Content-Length: {}\r\n"
                    "Content-Type: application/vscode-jsonrpc; charset=utf8\r\n\r\n"
                    "{}".format(content_length, body))
                self._wfile.write(response.encode('utf-8'))
                await self._wfile.drain()
            except Exception:  # pylint: disable=broad-except
                log.exception("Failed to write message to output file %s",
                              message)


from pathlib import Path


async def pyright_main(debug_proj_root: str, debug_code: str):
    python_path = sys.executable
    messages = [{
        'jsonrpc': '2.0',
        'id': 0,
        'method': 'initialize',
        'params': {
            'processId': None,
            'clientInfo': {
                'name': 'Monaco',
                'version': '1.76.0'
            },
            'locale': 'en-US',
            'rootPath': None,
            'rootUri': None,
            'capabilities': {
                'workspace': {
                    'applyEdit': True,
                    'workspaceEdit': {
                        'documentChanges': True,
                        'resourceOperations': ['create', 'rename', 'delete'],
                        'failureHandling': 'textOnlyTransactional',
                        'normalizesLineEndings': True,
                        'changeAnnotationSupport': {
                            'groupsOnLabel': True
                        }
                    },
                    'configuration': True,
                    'codeLens': {
                        'refreshSupport': True
                    },
                    'executeCommand': {
                        'dynamicRegistration': True
                    },
                    'didChangeConfiguration': {
                        'dynamicRegistration': True
                    },
                    'workspaceFolders': True,
                    'semanticTokens': {
                        'refreshSupport': True
                    },
                    'inlayHint': {
                        'refreshSupport': True
                    },
                    'diagnostics': {
                        'refreshSupport': True
                    }
                },
                'textDocument': {
                    'publishDiagnostics': {
                        'relatedInformation': True,
                        'versionSupport': False,
                        'tagSupport': {
                            'valueSet': [1, 2]
                        },
                        'codeDescriptionSupport': True,
                        'dataSupport': True
                    },
                    'synchronization': {
                        'dynamicRegistration': True
                    },
                    'completion': {
                        'dynamicRegistration': True,
                        'contextSupport': True,
                        'completionItem': {
                            'snippetSupport': True,
                            'commitCharactersSupport': True,
                            'documentationFormat': ['markdown', 'plaintext'],
                            'deprecatedSupport': True,
                            'preselectSupport': True,
                            'tagSupport': {
                                'valueSet': [1]
                            },
                            'insertReplaceSupport': True,
                            'resolveSupport': {
                                'properties': [
                                    'documentation', 'detail',
                                    'additionalTextEdits'
                                ]
                            },
                            'insertTextModeSupport': {
                                'valueSet': [1, 2]
                            },
                            'labelDetailsSupport': True
                        },
                        'insertTextMode': 2,
                        'completionItemKind': {
                            'valueSet': [
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
                            ]
                        },
                        'completionList': {
                            'itemDefaults': [
                                'commitCharacters', 'editRange',
                                'insertTextFormat', 'insertTextMode'
                            ]
                        }
                    },
                    'hover': {
                        'dynamicRegistration': True,
                        'contentFormat': ['markdown', 'plaintext']
                    },
                    'signatureHelp': {
                        'dynamicRegistration': True,
                        'signatureInformation': {
                            'documentationFormat': ['markdown', 'plaintext'],
                            'parameterInformation': {
                                'labelOffsetSupport': True
                            },
                            'activeParameterSupport': True
                        },
                        'contextSupport': True
                    },
                    'definition': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'references': {
                        'dynamicRegistration': True
                    },
                    'documentHighlight': {
                        'dynamicRegistration': True
                    },
                    'documentSymbol': {
                        'dynamicRegistration': True,
                        'symbolKind': {
                            'valueSet': [
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
                            ]
                        },
                        'hierarchicalDocumentSymbolSupport': True,
                        'tagSupport': {
                            'valueSet': [1]
                        },
                        'labelSupport': True
                    },
                    'codeAction': {
                        'dynamicRegistration': True,
                        'isPreferredSupport': True,
                        'disabledSupport': True,
                        'dataSupport': True,
                        'resolveSupport': {
                            'properties': ['edit']
                        },
                        'codeActionLiteralSupport': {
                            'codeActionKind': {
                                'valueSet': [
                                    '', 'quickfix', 'refactor',
                                    'refactor.extract', 'refactor.inline',
                                    'refactor.rewrite', 'source',
                                    'source.organizeImports'
                                ]
                            }
                        },
                        'honorsChangeAnnotations': False
                    },
                    'codeLens': {
                        'dynamicRegistration': True
                    },
                    'formatting': {
                        'dynamicRegistration': True
                    },
                    'rangeFormatting': {
                        'dynamicRegistration': True
                    },
                    'onTypeFormatting': {
                        'dynamicRegistration': True
                    },
                    'rename': {
                        'dynamicRegistration': True,
                        'prepareSupport': True,
                        'prepareSupportDefaultBehavior': 1,
                        'honorsChangeAnnotations': True
                    },
                    'documentLink': {
                        'dynamicRegistration': True,
                        'tooltipSupport': True
                    },
                    'typeDefinition': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'implementation': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'colorProvider': {
                        'dynamicRegistration': True
                    },
                    'foldingRange': {
                        'dynamicRegistration': True,
                        'rangeLimit': 5000,
                        'lineFoldingOnly': True,
                        'foldingRangeKind': {
                            'valueSet': ['comment', 'imports', 'region']
                        },
                        'foldingRange': {
                            'collapsedText': False
                        }
                    },
                    'declaration': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'selectionRange': {
                        'dynamicRegistration': True
                    },
                    'semanticTokens': {
                        'dynamicRegistration':
                        True,
                        'tokenTypes': [
                            'namespace', 'type', 'class', 'enum', 'interface',
                            'struct', 'typeParameter', 'parameter', 'variable',
                            'property', 'enumMember', 'event', 'function',
                            'method', 'macro', 'keyword', 'modifier',
                            'comment', 'string', 'number', 'regexp',
                            'operator', 'decorator'
                        ],
                        'tokenModifiers': [
                            'declaration', 'definition', 'readonly', 'static',
                            'deprecated', 'abstract', 'async', 'modification',
                            'documentation', 'defaultLibrary'
                        ],
                        'formats': ['relative'],
                        'requests': {
                            'range': True,
                            'full': {
                                'delta': True
                            }
                        },
                        'multilineTokenSupport':
                        False,
                        'overlappingTokenSupport':
                        False,
                        'serverCancelSupport':
                        True,
                        'augmentsSyntaxTokens':
                        True
                    },
                    'linkedEditingRange': {
                        'dynamicRegistration': True
                    },
                    'inlayHint': {
                        'dynamicRegistration': True,
                        'resolveSupport': {
                            'properties': [
                                'tooltip', 'textEdits', 'label.tooltip',
                                'label.location', 'label.command'
                            ]
                        }
                    },
                    'diagnostic': {
                        'dynamicRegistration': True,
                        'relatedDocumentSupport': False
                    }
                },
                'window': {
                    'showMessage': {
                        'messageActionItem': {
                            'additionalPropertiesSupport': True
                        }
                    },
                    'showDocument': {
                        'support': True
                    }
                },
                'general': {
                    'staleRequestSupport': {
                        'cancel':
                        True,
                        'retryOnContentModified': [
                            'textDocument/semanticTokens/full',
                            'textDocument/semanticTokens/range',
                            'textDocument/semanticTokens/full/delta'
                        ]
                    },
                    'regularExpressions': {
                        'engine': 'ECMAScript',
                        'version': 'ES2020'
                    },
                    'markdown': {
                        'parser': 'marked',
                        'version': '1.1.0'
                    },
                    'positionEncodings': ['utf-16']
                }
            },
            'trace': 'off',
            'workspaceFolders': None
        }
    }, {
        'jsonrpc': '2.0',
        'method': 'initialized',
        'params': {}
    }, {
        'jsonrpc': '2.0',
        'method': 'textDocument/didOpen',
        'params': {
            'textDocument': {
                'uri': 'file:///default',
                'languageId': 'python',
                'version': 1,
                'text': ''
            }
        }
    }, {
        'jsonrpc':
        '2.0',
        'id':
        0,
        'result': [{
            'analysis': {
                'extraPaths': [debug_proj_root],
                'logLevel': "Trace",
                'pythonPath': python_path
            }
        }]
    }, {
        'jsonrpc':
        '2.0',
        'id':
        1,
        'result': [{
            'extraPaths': [debug_proj_root],
            'logLevel': "Trace",
            'pythonPath': python_path
        }]
    }, {
        'jsonrpc': '2.0',
        'id': 2,
        'result': [None]
    }, {
        'jsonrpc': '2.0',
        'method': 'textDocument/didOpen',
        'params': {
            'textDocument': {
                'uri':
                'file:///%3C__tensorpc_inmemory_fname-01-basic/1.1-Hello%20World.md-1%3E',
                'languageId':
                'python',
                'version':
                1,
                'text':
                debug_code
            }
        }
    }, {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'textDocument/codeAction',
        'params': {
            'textDocument': {
                'uri':
                'file:///%3C__tensorpc_inmemory_fname-01-basic/1.1-Hello%20World.md-1%3E'
            },
            'range': {
                'start': {
                    'line': 0,
                    'character': 0
                },
                'end': {
                    'line': 0,
                    'character': 0
                }
            },
            'context': {
                'diagnostics': [],
                'triggerKind': 2
            }
        }
    }, {
        'jsonrpc': '2.0',
        'id': 2,
        'method': 'textDocument/codeAction',
        'params': {
            'textDocument': {
                'uri':
                'file:///%3C__tensorpc_inmemory_fname-01-basic/1.1-Hello%20World.md-1%3E'
            },
            'range': {
                'start': {
                    'line': 0,
                    'character': 0
                },
                'end': {
                    'line': 0,
                    'character': 0
                }
            },
            'context': {
                'diagnostics': [],
                'triggerKind': 2
            }
        }
    }, {
        'jsonrpc': '2.0',
        'id': 3,
        'method': 'textDocument/hover',
        'params': {
            'textDocument': {
                'uri':
                'file:///%3C__tensorpc_inmemory_fname-01-basic/1.1-Hello%20World.md-1%3E'
            },
            'position': {
                'line': 0,
                'character': 17
            }
        }
    }]

    aproc = await asyncio.create_subprocess_exec("python",
                                                 str(Path(__file__).resolve()),
                                                 "--is_ls",
                                                 env=os.environ,
                                                 stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE)
    assert aproc.stdout is not None
    assert aproc.stdin is not None
    await asyncio.sleep(1)
    # Create a writer that formats json messages with the correct LSP headers
    writer = AsyncJsonRpcStreamWriter(aproc.stdin)
    reader = AsyncJsonRpcStreamReader(aproc.stdout)

    async def cosumer(msg):
        print("[JSONRPC OUT]", msg)

    task = asyncio.create_task(reader.listen(cosumer))

    for msg in messages:
        print("[JSONRPC IN]", msg)
        await writer.write(msg)
        await asyncio.sleep(0.2)
    await asyncio.sleep(3.0)


def main_split():
    argv = sys.argv
    if "--is_ls" in argv:
        # we use pyright from pypi, you may need to change command to use your npm pyright
        from pyright.langserver import run
        run("--stdio")
    else:
        debug_proj_path = "/root/debug_project"
        debug_code = 'import pyright_debug_module\n'
        asyncio.run(pyright_main(debug_proj_path, debug_code))


if __name__ == "__main__":
    main_split()
