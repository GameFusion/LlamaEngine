
int main_vision(const char *prompt, const char *image);

bool hasVision();
bool generateVision(int session_id, const std::string &input_prompt, void (*callback)(const char*, void *userData), void *userData);
