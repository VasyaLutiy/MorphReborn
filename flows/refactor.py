from pysyun.conversation.flow.console_bot import ConsoleBot


class RefactorBot(ConsoleBot):

    def __init__(self, token):
        super().__init__(token)

    def build_state_machine(self, builder):
        main_menu_transition = self.build_menu_response_transition(
            "Morph your project",
            [["Code review", "File structure review"], ["Generate"]])

        return builder \
            .edge(
                "/start",
                "/graph",
                "/graph",
                on_transition=self.build_graphviz_response_transition()) \
            .edge("/start", "/start", "/start", on_transition=main_menu_transition)
