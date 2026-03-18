from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Label, ListItem, ListView, Static


@dataclass
class MidiEvent:
    step: int
    note: str
    length: int = 1
    velocity: int = 100
    channel: int = 1


@dataclass
class Track:
    name: str
    channel: int = 1
    mute: bool = False
    solo: bool = False
    events: List[MidiEvent] = field(default_factory=list)


class StatusBar(Static):
    def set_text(self, text: str) -> None:
        self.update(text)


class AddTrackModal(ModalScreen[str | None]):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("New Track Name"),
            Input(placeholder="e.g. Bassline", id="track_name_input"),
            Label("Enter to confirm, Esc to cancel"),
            id="add_track_modal",
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


class MidiWorkspace(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #left_panel {
        width: 28;
        min-width: 24;
        border: solid $primary;
        padding: 1;
    }

    #center_panel {
        width: 1fr;
        border: solid $accent;
        padding: 1;
    }

    #right_panel {
        width: 28;
        min-width: 24;
        border: solid $warning;
        padding: 1;
    }

    #status {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text;
        padding: 0 1;
    }

    #track_list {
        height: 1fr;
    }

    #event_grid {
        height: 1fr;
    }

    #inspector {
        height: 1fr;
    }

    .panel_title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "add_track", "Add Track"),
        ("d", "delete_track", "Delete Track"),
        ("tab", "focus_next", "Next Pane"),
        ("shift+tab", "focus_previous", "Prev Pane"),
        ("space", "toggle_play", "Play/Stop"),
        ("i", "insert_event", "Insert Note"),
        ("x", "delete_event", "Delete Note"),
        ("m", "toggle_mute", "Mute"),
        ("s", "toggle_solo", "Solo"),
    ]

    current_track_index = reactive(0)
    is_playing = reactive(False)
    current_step = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self.tracks: List[Track] = [
            Track(
                name="Lead",
                channel=1,
                events=[
                    MidiEvent(step=0, note="C5", length=2, velocity=110),
                    MidiEvent(step=4, note="E5", length=2, velocity=108),
                    MidiEvent(step=8, note="G5", length=4, velocity=115),
                ],
            ),
            Track(
                name="Bass",
                channel=2,
                events=[
                    MidiEvent(step=0, note="C2", length=4, velocity=120),
                    MidiEvent(step=8, note="G2", length=4, velocity=120),
                ],
            ),
            Track(name="Drums", channel=10),
        ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Static("Tracks", classes="panel_title"),
                ListView(id="track_list"),
                id="left_panel",
            ),
            Vertical(
                Static("Piano Roll / Step Grid", classes="panel_title"),
                DataTable(id="event_grid"),
                id="center_panel",
            ),
            Vertical(
                Static("Inspector", classes="panel_title"),
                Static(id="inspector"),
                id="right_panel",
            ),
            id="body",
        )
        yield StatusBar("Ready", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_track_list()
        self._setup_grid()
        self._refresh_grid()
        self._refresh_inspector()
        self._set_status("Workspace loaded")

    def _track_list(self) -> ListView:
        return self.query_one("#track_list", ListView)

    def _grid(self) -> DataTable:
        return self.query_one("#event_grid", DataTable)

    def _inspector(self) -> Static:
        return self.query_one("#inspector", Static)

    def _status(self) -> StatusBar:
        return self.query_one("#status", StatusBar)

    def _set_status(self, text: str) -> None:
        self._status().set_text(text)

    def _refresh_track_list(self) -> None:
        view = self._track_list()
        view.clear()
        for i, track in enumerate(self.tracks):
            flags = []
            if track.mute:
                flags.append("M")
            if track.solo:
                flags.append("S")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            marker = "▶ " if i == self.current_track_index else "  "
            view.append(ListItem(Label(f"{marker}{track.name}  ch:{track.channel}{suffix}")))
        if self.tracks:
            view.index = self.current_track_index

    def _setup_grid(self) -> None:
        grid = self._grid()
        grid.clear(columns=True)
        grid.cursor_type = "cell"
        grid.zebra_stripes = True
        columns = ["Note"] + [str(i) for i in range(16)]
        for col in columns:
            grid.add_column(col, width=6 if col == "Note" else 4)

    def _refresh_grid(self) -> None:
        grid = self._grid()
        grid.clear()
        if not self.tracks:
            return

        current = self.tracks[self.current_track_index]
        note_rows = ["C6", "B5", "A5", "G5", "F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4", "B3", "A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2", "F2", "E2", "D2", "C2"]

        event_map = {(event.note, event.step): event for event in current.events}

        for note in note_rows:
            cells = [note]
            for step in range(16):
                if (note, step) in event_map:
                    event = event_map[(note, step)]
                    cells.append("■" if event.length == 1 else f"{event.length}")
                else:
                    cells.append("·")
            grid.add_row(*cells)

    def _refresh_inspector(self) -> None:
        if not self.tracks:
            self._inspector().update("No track selected")
            return

        track = self.tracks[self.current_track_index]
        text = (
            f"Track: {track.name}\n"
            f"Channel: {track.channel}\n"
            f"Mute: {track.mute}\n"
            f"Solo: {track.solo}\n"
            f"Events: {len(track.events)}\n\n"
            f"Transport\n"
            f"Playing: {self.is_playing}\n"
            f"Step: {self.current_step}\n\n"
            f"Template behavior\n"
            f"- A: add track\n"
            f"- D: delete track\n"
            f"- I: insert note at cursor\n"
            f"- X: delete note at cursor\n"
            f"- M/S: mute/solo\n"
            f"- Space: play toggle"
        )
        self._inspector().update(text)

    def action_add_track(self) -> None:
        self.push_screen(AddTrackModal(), self._add_track_result)

    def _add_track_result(self, name: str | None) -> None:
        if not name:
            self._set_status("Add track cancelled")
            return
        self.tracks.append(Track(name=name, channel=min(len(self.tracks) + 1, 16)))
        self.current_track_index = len(self.tracks) - 1
        self._refresh_track_list()
        self._refresh_grid()
        self._refresh_inspector()
        self._set_status(f"Added track: {name}")

    def action_delete_track(self) -> None:
        if not self.tracks:
            return
        removed = self.tracks.pop(self.current_track_index)
        if self.current_track_index >= len(self.tracks):
            self.current_track_index = max(0, len(self.tracks) - 1)
        self._refresh_track_list()
        self._refresh_grid()
        self._refresh_inspector()
        self._set_status(f"Deleted track: {removed.name}")

    def action_toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        self._set_status("Playing" if self.is_playing else "Stopped")
        self._refresh_inspector()

    def action_toggle_mute(self) -> None:
        if not self.tracks:
            return
        track = self.tracks[self.current_track_index]
        track.mute = not track.mute
        self._refresh_track_list()
        self._refresh_inspector()
        self._set_status(f"Mute {track.name}: {track.mute}")

    def action_toggle_solo(self) -> None:
        if not self.tracks:
            return
        track = self.tracks[self.current_track_index]
        track.solo = not track.solo
        self._refresh_track_list()
        self._refresh_inspector()
        self._set_status(f"Solo {track.name}: {track.solo}")

    def action_insert_event(self) -> None:
        if not self.tracks:
            return
        grid = self._grid()
        coord = grid.cursor_coordinate
        if coord is None:
            self._set_status("No grid cell selected")
            return

        row, column = coord.row, coord.column
        if column == 0:
            self._set_status("Select a step column, not the note label")
            return

        note = str(grid.get_cell_at((row, 0)))
        step = column - 1
        track = self.tracks[self.current_track_index]

        for event in track.events:
            if event.note == note and event.step == step:
                self._set_status(f"Event already exists at {note} step {step}")
                return

        track.events.append(MidiEvent(step=step, note=note, length=1, velocity=100, channel=track.channel))
        self._refresh_grid()
        self._refresh_inspector()
        self._set_status(f"Inserted {note} at step {step}")

    def action_delete_event(self) -> None:
        if not self.tracks:
            return
        grid = self._grid()
        coord = grid.cursor_coordinate
        if coord is None:
            self._set_status("No grid cell selected")
            return

        row, column = coord.row, coord.column
        if column == 0:
            self._set_status("Select an event cell")
            return

        note = str(grid.get_cell_at((row, 0)))
        step = column - 1
        track = self.tracks[self.current_track_index]
        before = len(track.events)
        track.events = [e for e in track.events if not (e.note == note and e.step == step)]
        after = len(track.events)

        self._refresh_grid()
        self._refresh_inspector()
        if before == after:
            self._set_status(f"No event at {note} step {step}")
        else:
            self._set_status(f"Deleted event at {note} step {step}")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "track_list":
            return
        if event.list_view.index is None:
            return
        self.current_track_index = event.list_view.index
        self._refresh_track_list()
        self._refresh_grid()
        self._refresh_inspector()
        self._set_status(f"Selected track: {self.tracks[self.current_track_index].name}")


if __name__ == "__main__":
    MidiWorkspace().run()
