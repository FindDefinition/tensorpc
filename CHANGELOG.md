# Changelog

# [0.15.0] - 2025-02-xx
### Added 
- add compute flow v2 with nested flow support, schedule node support and more.

# [0.14.0] - 2025-01-24
### Added
- add task wrapper based on asyncssh.
- add data model based components for `MVC` design pattern.
- add terminal component.

# [0.13.0] - 2024-11-2x
### Added 
- add pytorch `torch.export` flow graph support
- BREAKING CHANGE: change implementation of component event handling, you must use devflow v0.13.x to work with this version.

# [0.12.0] - 2024-10-2x
### Added 
- add remote component support.
- add `tensorpc.dbg` package for debug-with-ui purpose.

# [0.11.19] - 2024-07-23
### Added 
- add a special handle type in compute flow.
### Fixed
- fix a serious bug when user use nested layout-change method.
- fix watchdog problem in mac os
- fix exception handling in ctrl loop.

# [0.11.18] - 2024-07-22
### Changed
- change overflow in compute flow node
- refine controlled loop impl

# [0.11.17] - 2024-07-19
### Added
- add controlled loop that can make loop in sync functions controllable via `ControlledLoop` component.
- add label render support for points
### Fixed
- fix caller lineno in call tracer
- fix small bug in compute flow
- fix some bug in shutdown logic

# [0.11.16] - 2024-07-17
### Added
- add node context for compute flow, user can access node id in node layout callback or compute.
### Changed
- file watchdog now enable debounce (0.1s) by default to avoid incomplete write.

# [0.11.15] - 2024-07-16
### Changed 
- refine some app feature impl
- better vscode trace info
- change base component file name and many import stmts.

# [0.11.14] - 2024-07-15
### Added 
- add vscode trace tree query, user can run trace and send result to vscode. requires vscode-tensorpc-bridge extension >= 0.2.0

# [0.11.13] - 2024-07-14
### Changed 
- change tooltip prop for some component
- change keybinding format in action for monaco editor
### Added 
- add cache input to compute flow node
- add context menu per node in flowui
- add process pool executor support for compute flow custom node

# [0.11.12] - 2024-07-10
### Fixed 
- fix some bug in compute flow
- fix bug in annocore

# [0.11.11] - 2024-07-07
### Fixed 
- fix wrong schedule result in a certain case in compute flow
### Added 
- add TensorViewer in compute flow

# [0.11.10] - 2024-07-04
### Fixed 
- fix some compute flow small bug
### Added 
- add duration display to compute flow node
- add style override for node style

# [0.11.9] - 2024-07-03
### Fixed 
- fix bugs of templates in compute flow
### Added 
- add a template manager for delete in compute flow

# [0.11.8] - 2024-07-02
### Changed 
- change monaco editor save event format
### Fixed 
- fix some bug in compute flow

# [0.11.7] - 2024-07-01
### Added 
- add actions to monaco editor
- add pane context menu item update to simple flow
### Changed
- refine some compute flow logic 
### Fixed 
- fix python 3.8 compatibility issue

# [0.11.6] - 2024-06-30
### Fixed 
- fix some bug

# [0.11.5] - 2024-06-20
### Added 
- add compute flow
### Fixed 
- fix lots of bugs

# [0.11.4] - 2024-06-10
### Fixed 
- fix shell problem in mac os
- add zsh default shell support

# [0.11.3] - 2024-05-26
### Changed
- change valid connect logic

# [0.11.2] - 2024-05-26
### Fixed
- refactor some code

# [0.11.1] - 2024-05-25
### Fixed
- fix some flow bug

# [0.11.0] - 2024-05-24
### Added 
- add flow graph based on xyflow. must be used on dock mode.

# [0.10.7] - 2024-05-11
### Changed 
- change year in license
### Added
- add option to attach metadata when app runs

# [0.10.6] - 2024-04-27
### Changed
- change all legacy server events to standard api

# [0.10.5] - 2024-04-26
### Fixed
- fix lazy init services don't work in previous release

# [0.10.4] - 2024-04-25
### Added
- add basic support for standard server event
- add simple bg server

# [0.10.3] - 2024-04-18
### Fixed
- fix port in meta when use dynamic port

# [0.10.2] - 2024-04-18
### Fixed
- fix bug in inspector
### Changed
- change default grpc option: reuseport = False

# [0.10.1] - 2024-04-08
### Fixed
- fix a small bug with newest asyncssh

# [0.10.0] - 2024-xx-xx
### Added
- add changelog for tensorpc package
- add support for devflow v0.10.x (tree view of apps)