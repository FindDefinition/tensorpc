# ---------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------------------------------------------

# Prevent the script recursing when setting up
if [[ -n "${VSCODE_SHELL_INTEGRATION:-}" ]]; then
	builtin return
fi

VSCODE_SHELL_INTEGRATION=1
VSCODE_INJECTION=1
# enable shell login to get same environment as manual ssh 
VSCODE_SHELL_LOGIN=1
# Run relevant rc/profile only if shell integration has been injected, not when run manually
if [ "$VSCODE_INJECTION" == "1" ]; then
	if [ -z "$VSCODE_SHELL_LOGIN" ]; then
		if [ -r ~/.bashrc ]; then
			. ~/.bashrc
		fi
	else
		# Imitate -l because --init-file doesn't support it:
		# run the first of these files that exists
		if [ -r /etc/profile ]; then
			. /etc/profile
		fi
		# execute the first that exists
		if [ -r ~/.bash_profile ]; then
			. ~/.bash_profile
		elif [ -r ~/.bash_login ]; then
			. ~/.bash_login
		elif [ -r ~/.profile ]; then
			. ~/.profile
		fi
		builtin unset VSCODE_SHELL_LOGIN

		# Apply any explicit path prefix (see #99878)
		if [ -n "${VSCODE_PATH_PREFIX:-}" ]; then
			export PATH=$VSCODE_PATH_PREFIX$PATH
			builtin unset VSCODE_PATH_PREFIX
		fi
	fi
	builtin unset VSCODE_INJECTION
fi

if [ -z "$VSCODE_SHELL_INTEGRATION" ]; then
	builtin return
fi

# Apply EnvironmentVariableCollections if needed
if [ -n "${VSCODE_ENV_REPLACE:-}" ]; then
	IFS=':' read -ra ADDR <<< "$VSCODE_ENV_REPLACE"
	for ITEM in "${ADDR[@]}"; do
		VARNAME="$(echo $ITEM | cut -d "=" -f 1)"
		VALUE="$(echo -e "$ITEM" | cut -d "=" -f 2-)"
		export $VARNAME="$VALUE"
	done
	builtin unset VSCODE_ENV_REPLACE
fi
if [ -n "${VSCODE_ENV_PREPEND:-}" ]; then
	IFS=':' read -ra ADDR <<< "$VSCODE_ENV_PREPEND"
	for ITEM in "${ADDR[@]}"; do
		VARNAME="$(echo $ITEM | cut -d "=" -f 1)"
		VALUE="$(echo -e "$ITEM" | cut -d "=" -f 2-)"
		export $VARNAME="$VALUE${!VARNAME}"
	done
	builtin unset VSCODE_ENV_PREPEND
fi
if [ -n "${VSCODE_ENV_APPEND:-}" ]; then
	IFS=':' read -ra ADDR <<< "$VSCODE_ENV_APPEND"
	for ITEM in "${ADDR[@]}"; do
		VARNAME="$(echo $ITEM | cut -d "=" -f 1)"
		VALUE="$(echo -e "$ITEM" | cut -d "=" -f 2-)"
		export $VARNAME="${!VARNAME}$VALUE"
	done
	builtin unset VSCODE_ENV_APPEND
fi

__vsc_get_trap() {
	# 'trap -p DEBUG' outputs a shell command like `trap -- '…shellcode…' DEBUG`.
	# The terms are quoted literals, but are not guaranteed to be on a single line.
	# (Consider a trap like $'echo foo\necho \'bar\'').
	# To parse, we splice those terms into an expression capturing them into an array.
	# This preserves the quoting of those terms: when we `eval` that expression, they are preserved exactly.
	# This is different than simply exploding the string, which would split everything on IFS, oblivious to quoting.
	builtin local -a terms
	builtin eval "terms=( $(trap -p "${1:-DEBUG}") )"
	#                    |________________________|
	#                            |
	#        \-------------------*--------------------/
	# terms=( trap  --  '…arbitrary shellcode…'  DEBUG )
	#        |____||__| |_____________________| |_____|
	#          |    |            |                |
	#          0    1            2                3
	#                            |
	#                   \--------*----/
	builtin printf '%s' "${terms[2]:-}"
}

__vsc_escape_value_fast() {
	builtin local LC_ALL=C out
	out=${1//\\/\\\\}
	out=${out//;/\\x3b}
	builtin printf '%s\n' "${out}"
}

# The property (P) and command (E) codes embed values which require escaping.
# Backslashes are doubled. Non-alphanumeric characters are converted to escaped hex.
__vsc_escape_value() {
	# If the input being too large, switch to the faster function
	if [ "${#1}" -ge 2000 ]; then
		__vsc_escape_value_fast "$1"
		builtin return
	fi

	# Process text byte by byte, not by codepoint.
	builtin local LC_ALL=C str="${1}" i byte token out=''

	for (( i=0; i < "${#str}"; ++i )); do
		byte="${str:$i:1}"
		# Escape backslashes, semi-colons specially, then special ASCII chars below space (0x20).
		# This is done in an unwrapped loop instead of using printf as the latter is very slow.
		if [ "$byte" = "\\" ]; then
			token="\\\\"
		elif [ "$byte" = ";" ]; then
			token="\\x3b"
		elif [ "$byte" = $'\x00' ]; then token="\\x00"
		elif [ "$byte" = $'\x01' ]; then token="\\x01"
		elif [ "$byte" = $'\x02' ]; then token="\\x02"
		elif [ "$byte" = $'\x03' ]; then token="\\x03"
		elif [ "$byte" = $'\x04' ]; then token="\\x04"
		elif [ "$byte" = $'\x05' ]; then token="\\x05"
		elif [ "$byte" = $'\x06' ]; then token="\\x06"
		elif [ "$byte" = $'\x07' ]; then token="\\x07"
		elif [ "$byte" = $'\x08' ]; then token="\\x08"
		elif [ "$byte" = $'\x09' ]; then token="\\x09"
		elif [ "$byte" = $'\x0a' ]; then token="\\x0a"
		elif [ "$byte" = $'\x0b' ]; then token="\\x0b"
		elif [ "$byte" = $'\x0c' ]; then token="\\x0c"
		elif [ "$byte" = $'\x0d' ]; then token="\\x0d"
		elif [ "$byte" = $'\x0e' ]; then token="\\x0e"
		elif [ "$byte" = $'\x0f' ]; then token="\\x0f"
		elif [ "$byte" = $'\x10' ]; then token="\\x10"
		elif [ "$byte" = $'\x11' ]; then token="\\x11"
		elif [ "$byte" = $'\x12' ]; then token="\\x12"
		elif [ "$byte" = $'\x13' ]; then token="\\x13"
		elif [ "$byte" = $'\x14' ]; then token="\\x14"
		elif [ "$byte" = $'\x15' ]; then token="\\x15"
		elif [ "$byte" = $'\x16' ]; then token="\\x16"
		elif [ "$byte" = $'\x17' ]; then token="\\x17"
		elif [ "$byte" = $'\x18' ]; then token="\\x18"
		elif [ "$byte" = $'\x19' ]; then token="\\x19"
		elif [ "$byte" = $'\x1a' ]; then token="\\x1a"
		elif [ "$byte" = $'\x1b' ]; then token="\\x1b"
		elif [ "$byte" = $'\x1c' ]; then token="\\x1c"
		elif [ "$byte" = $'\x1d' ]; then token="\\x1d"
		elif [ "$byte" = $'\x1e' ]; then token="\\x1e"
		elif [ "$byte" = $'\x1f' ]; then token="\\x1f"
		else
			token="$byte"
		fi

		out+="$token"
	done

	builtin printf '%s\n' "${out}"
}

# Send the IsWindows property if the environment looks like Windows
if [[ "$(uname -s)" =~ ^CYGWIN*|MINGW*|MSYS* ]]; then
	builtin printf '\e]784;P;IsWindows=True\a'
	__vsc_is_windows=1
else
	__vsc_is_windows=0
fi

# Allow verifying $BASH_COMMAND doesn't have aliases resolved via history when the right HISTCONTROL
# configuration is used
if [[ "$HISTCONTROL" =~ .*(erasedups|ignoreboth|ignoredups).* ]]; then
	__vsc_history_verify=0
else
	__vsc_history_verify=1
fi

__vsc_initialized=0
__vsc_original_PS1="$PS1"
__vsc_original_PS2="$PS2"
__vsc_custom_PS1=""
__vsc_custom_PS2=""
__vsc_in_command_execution="1"
__vsc_current_command=""

# It's fine this is in the global scope as it getting at it requires access to the shell environment
__vsc_nonce="$VSCODE_NONCE"
unset VSCODE_NONCE

# Report continuation prompt
builtin printf "\e]784;P;ContinuationPrompt=$(echo "$PS2" | sed 's/\x1b/\\\\x1b/g')\a"

__vsc_report_prompt() {
	# HACK: Git bash is too slow at reporting the prompt, so skip for now
	if [ "$__vsc_is_windows" = "1" ]; then
		return
	fi

	# Expand the original PS1 similarly to how bash would normally
	# See https://stackoverflow.com/a/37137981 for technique
	if ((BASH_VERSINFO[0] >= 5 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4))); then
		__vsc_prompt=${__vsc_original_PS1@P}
	else
		__vsc_prompt=${__vsc_original_PS1}
	fi

	__vsc_prompt="$(builtin printf "%s" "${__vsc_prompt//[$'\001'$'\002']}")"
	builtin printf "\e]784;P;Prompt=%s\a" "$(__vsc_escape_value "${__vsc_prompt}")"
}

__vsc_prompt_start() {
	builtin printf '\e]784;A\a'
}

__vsc_prompt_end() {
	builtin printf '\e]784;B\a'
}

__vsc_update_cwd() {
	if [ "$__vsc_is_windows" = "1" ]; then
		__vsc_cwd="$(cygpath -m "$PWD")"
	else
		__vsc_cwd="$PWD"
	fi
	builtin printf '\e]784;P;Cwd=%s\a' "$(__vsc_escape_value "$__vsc_cwd")"
}

__vsc_command_output_start() {
	if [[ -z "$__vsc_first_prompt" ]]; then
		builtin return
	fi
	builtin printf '\e]784;E;%s;%s\a' "$(__vsc_escape_value "${__vsc_current_command}")" $__vsc_nonce
	builtin printf '\e]784;C\a'
}

__vsc_continuation_start() {
	builtin printf '\e]784;F\a'
}

__vsc_continuation_end() {
	builtin printf '\e]784;G\a'
}

__vsc_command_complete() {
	if [[ -z "$__vsc_first_prompt" ]]; then
		builtin return
	fi
	if [ "$__vsc_current_command" = "" ]; then
		builtin printf '\e]784;D\a'
	else
		builtin printf '\e]784;D;%s\a' "$__vsc_status"
	fi
	__vsc_update_cwd
}
__vsc_update_prompt() {
	# in command execution
	if [ "$__vsc_in_command_execution" = "1" ]; then
		# Wrap the prompt if it is not yet wrapped, if the PS1 changed this this was last set it
		# means the user re-exported the PS1 so we should re-wrap it
		if [[ "$__vsc_custom_PS1" == "" || "$__vsc_custom_PS1" != "$PS1" ]]; then
			__vsc_original_PS1=$PS1
			__vsc_custom_PS1="\[$(__vsc_prompt_start)\]$__vsc_original_PS1\[$(__vsc_prompt_end)\]"
			PS1="$__vsc_custom_PS1"
		fi
		if [[ "$__vsc_custom_PS2" == "" || "$__vsc_custom_PS2" != "$PS2" ]]; then
			__vsc_original_PS2=$PS2
			__vsc_custom_PS2="\[$(__vsc_continuation_start)\]$__vsc_original_PS2\[$(__vsc_continuation_end)\]"
			PS2="$__vsc_custom_PS2"
		fi
		__vsc_in_command_execution="0"
	fi
}

__vsc_precmd() {
	__vsc_command_complete "$__vsc_status"
	__vsc_current_command=""
	__vsc_report_prompt
	__vsc_first_prompt=1
	__vsc_update_prompt
}

__vsc_preexec() {
	__vsc_initialized=1
	if [[ ! $BASH_COMMAND == __vsc_prompt* ]]; then
		# Use history if it's available to verify the command as BASH_COMMAND comes in with aliases
		# resolved
		if [ "$__vsc_history_verify" = "1" ]; then
			__vsc_current_command="$(builtin history 1 | sed 's/ *[0-9]* *//')"
		else
			__vsc_current_command=$BASH_COMMAND
		fi
	else
		__vsc_current_command=""
	fi
	__vsc_command_output_start
}

# remove all trap
# # trap | awk '{ print $NF }' | while read SIG ; do trap - $SIG ; done
# echo $(compgen -A signal)
# trap - $(compgen -A signal)
trap - DEBUG

# Debug trapping/preexec inspired by starship (ISC)
if [[ -n "${bash_preexec_imported:-}" ]]; then
	__vsc_preexec_only() {
		if [ "$__vsc_in_command_execution" = "0" ]; then
			__vsc_in_command_execution="1"
			__vsc_preexec
		fi
	}
	precmd_functions+=(__vsc_prompt_cmd)
	preexec_functions+=(__vsc_preexec_only)
else
	__vsc_dbg_trap="$(__vsc_get_trap DEBUG)"
	if [[ -z "$__vsc_dbg_trap" ]]; then
		__vsc_preexec_only() {
			if [ "$__vsc_in_command_execution" = "0" ]; then
				__vsc_in_command_execution="1"
				__vsc_preexec
			fi
		}
		trap '__vsc_preexec_only "$_"' DEBUG
	elif [[ "$__vsc_dbg_trap" != '__vsc_preexec "$_"' && "$__vsc_dbg_trap" != '__vsc_preexec_all "$_"' ]]; then
		__vsc_preexec_all() {
			if [ "$__vsc_in_command_execution" = "0" ]; then
				__vsc_in_command_execution="1"
				__vsc_preexec
				builtin eval "${__vsc_dbg_trap}"
			fi
		}
		trap '__vsc_preexec_all "$_"' DEBUG
	fi
fi

__vsc_update_prompt

__vsc_restore_exit_code() {
	return "$1"
}

__vsc_prompt_cmd_original() {
	__vsc_status="$?"
	__vsc_restore_exit_code "${__vsc_status}"
	# Evaluate the original PROMPT_COMMAND similarly to how bash would normally
	# See https://unix.stackexchange.com/a/672843 for technique
	local cmd
	for cmd in "${__vsc_original_prompt_command[@]}"; do
		eval "${cmd:-}"
	done
	__vsc_precmd
}

__vsc_prompt_cmd() {
	__vsc_status="$?"
	__vsc_precmd
}

# PROMPT_COMMAND arrays and strings seem to be handled the same (handling only the first entry of
# the array?)
__vsc_original_prompt_command=${PROMPT_COMMAND:-}

if [[ -z "${bash_preexec_imported:-}" ]]; then
	if [[ -n "${__vsc_original_prompt_command:-}" && "${__vsc_original_prompt_command:-}" != "__vsc_prompt_cmd" ]]; then
		PROMPT_COMMAND=__vsc_prompt_cmd_original
	else
		PROMPT_COMMAND=__vsc_prompt_cmd
	fi
fi
export TERM=xterm-256color
export TENSORPC_SSH_CURRENT_PID=$$
export HISTCONTROL=ignorespace