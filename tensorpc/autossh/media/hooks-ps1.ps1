# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------------------------------------------

# Prevent installing more than once per session
if (Test-Path variable:global:__VSCodeOriginalPrompt) {
	return;
}

# Disable shell integration when the language mode is restricted
if ($ExecutionContext.SessionState.LanguageMode -ne "FullLanguage") {
	return;
}

$Global:__VSCodeOriginalPrompt = $function:Prompt

$Global:__LastHistoryId = -1

function Global:__VSCode-Get-LastExitCode {
	if ($? -eq $True) {
		return 0
	}
	# TODO: Should we just return a string instead?
	return -1
}

function Global:Prompt() {
	$LastExitCode = $(__VSCode-Get-LastExitCode);
	$LastHistoryEntry = $(Get-History -Count 1)
	# Skip finishing the command if the first command has not yet started
	if ($Global:__LastHistoryId -ne -1) {
		if ($LastHistoryEntry.Id -eq $Global:__LastHistoryId) {
			# Don't provide a command line or exit code if there was no history entry (eg. ctrl+c, enter on no command)
			$Result  = "`e]784;E`a"
			$Result += "`e]784;D`a"
		} else {
			# Command finished command line
			# OSC 784 ; A ; <CommandLine?> ST
			$Result  = "`e]784;E;"
			# Sanitize the command line to ensure it can get transferred to the terminal and can be parsed
			# correctly. This isn't entirely safe but good for most cases, it's important for the Pt parameter
			# to only be composed of _printable_ characters as per the spec.
			if ($LastHistoryEntry.CommandLine) {
				$CommandLine = $LastHistoryEntry.CommandLine
			} else {
				$CommandLine = ""
			}
			$Result += $CommandLine.Replace("\", "\\").Replace("`n", "\x0a").Replace(";", "\x3b")
			$Result += "`a"
			# Command finished exit code
			# OSC 784 ; D [; <ExitCode>] ST
			$Result += "`e]784;D;$LastExitCode`a"
		}
	}
	# Prompt started
	# OSC 784 ; A ST
	$Result += "`e]784;A`a"
	# Current working directory
	# OSC 784 ; <Property>=<Value> ST
	$Result += if($pwd.Provider.Name -eq 'FileSystem'){"`e]784;P;Cwd=$($pwd.ProviderPath)`a"}
	# Write original prompt
	$Result += $Global:__VSCodeOriginalPrompt.Invoke()
	# Write command started
	$Result += "`e]784;B`a"
	$Global:__LastHistoryId = $LastHistoryEntry.Id
	return $Result
}

# Only send the command executed sequence when PSReadLine is loaded, if not shell integration should
# still work thanks to the command line sequence
if (Get-Module -Name PSReadLine) {
	$__VSCodeOriginalPSConsoleHostReadLine = $function:PSConsoleHostReadLine
	function Global:PSConsoleHostReadLine {
		$tmp = $__VSCodeOriginalPSConsoleHostReadLine.Invoke()
		# Write command executed sequence directly to Console to avoid the new line from Write-Host
		[Console]::Write("`e]784;C`a")
		$tmp
	}
}

# Set IsWindows property
[Console]::Write("`e]784;P;IsWindows=$($IsWindows)`a")

# Set always on key handlers which map to default VS Code keybindings
function Set-MappedKeyHandler {
	param ([string[]] $Chord, [string[]]$Sequence)
	$Handler = $(Get-PSReadLineKeyHandler -Chord $Chord | Select-Object -First 1)
	if ($Handler) {
		Set-PSReadLineKeyHandler -Chord $Sequence -Function $Handler.Function
	}
}
function Set-MappedKeyHandlers {
	Set-MappedKeyHandler -Chord Ctrl+Spacebar -Sequence 'F12,a'
	Set-MappedKeyHandler -Chord Alt+Spacebar -Sequence 'F12,b'
	Set-MappedKeyHandler -Chord Shift+Enter -Sequence 'F12,c'
	Set-MappedKeyHandler -Chord Shift+End -Sequence 'F12,d'
}
Set-MappedKeyHandlers
