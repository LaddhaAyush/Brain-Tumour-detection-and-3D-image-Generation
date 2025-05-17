package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	cohere "github.com/cohere-ai/cohere-go/v2"
	client "github.com/cohere-ai/cohere-go/v2/client"
)

// BrainTumorBot represents the AI bot for brain tumor information
type BrainTumorBot struct {
	client         *client.Client
	medicalSources []string
	knowledgeBase  []map[string]string
	chatHistory    []cohere.ChatMessage
	mu             sync.Mutex
}

// TumorContext holds information about the patient's tumor
type TumorContext struct {
	Type     string  `json:"type"`
	Location string  `json:"location"`
	Size     float64 `json:"size"`
}

// NewBrainTumorBot creates a new instance of BrainTumorBot
func NewBrainTumorBot(apiKey string) *BrainTumorBot {
	return &BrainTumorBot{
		client: client.NewClient(client.WithToken(apiKey)),
		medicalSources: []string{
			"https://www.cancer.gov/types/brain",
			"https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/",
			"https://www.cancer.org/cancer/brain-spinal-cord-tumors-adults/",
			"https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor",
			"https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Brain-Tumors",
		},
		knowledgeBase: make([]map[string]string, 0),
		chatHistory:   make([]cohere.ChatMessage, 0),
	}
}

// scrapeMedicalInfo scrapes medical information from a URL
func (bot *BrainTumorBot) scrapeMedicalInfo(url string) (string, error) {
	client := &http.Client{
		Timeout: 10 * time.Second,
	}
	
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// TODO: Implement HTML parsing with a Go HTML parser
	// For now, return a placeholder
	return "Medical information from " + url, nil
}

// updateKnowledgeBase updates the knowledge base from medical sources
func (bot *BrainTumorBot) updateKnowledgeBase() error {
	bot.mu.Lock()
	defer bot.mu.Unlock()

	bot.knowledgeBase = make([]map[string]string, 0)
	for _, url := range bot.medicalSources {
		content, err := bot.scrapeMedicalInfo(url)
		if err != nil {
			log.Printf("Error scraping %s: %v", url, err)
			continue
		}
		bot.knowledgeBase = append(bot.knowledgeBase, map[string]string{
			"url":     url,
			"content": content,
		})
	}
	return nil
}

// Chat handles the chat interaction with the AI
func (bot *BrainTumorBot) Chat(ctx context.Context, userMessage string, tumorContext *TumorContext) (string, error) {
	// Update knowledge base
	if err := bot.updateKnowledgeBase(); err != nil {
		return "", err
	}

	// Build context from knowledge base
	var contextBuilder strings.Builder
	for _, item := range bot.knowledgeBase {
		contextBuilder.WriteString("Source: " + item["url"] + "\n" + item["content"] + "\n\n")
	}

	// Create system message
	systemMessage := cohere.ChatMessage{
		Role: "system",
		Content: "You are a medical expert specializing in brain tumors. " +
			"Provide clear, accurate, and specific information about brain tumors. " +
			"Always base your responses on current medical research and guidelines. " +
			"If you're unsure about something, acknowledge the limitations of your knowledge.",
	}

	// Add tumor context if available
	if tumorContext != nil {
		contextMessage := cohere.ChatMessage{
			Role: "system",
			Content: "Patient Context: The patient has a " + tumorContext.Type + " tumor",
		}
		if tumorContext.Location != "" {
			contextMessage.Content += " in the " + tumorContext.Location + " region"
		}
		if tumorContext.Size > 0 {
			contextMessage.Content += " of size " + string(tumorContext.Size)
		}
		bot.chatHistory = append(bot.chatHistory, contextMessage)
	}

	// Add user message
	userMessageObj := cohere.ChatMessage{
		Role:    "user",
		Content: userMessage,
	}
	bot.chatHistory = append(bot.chatHistory, userMessageObj)

	// Prepare chat request
	chatRequest := &cohere.ChatRequest{
		Messages:  bot.chatHistory,
		Context:   contextBuilder.String(),
		Model:     "command-r-plus",
		MaxTokens: 1000,
	}

	// Make API call
	resp, err := bot.client.Chat(ctx, chatRequest)
	if err != nil {
		return "", err
	}

	// Add assistant's response to chat history
	assistantMessage := cohere.ChatMessage{
		Role:    "assistant",
		Content: resp.Text,
	}
	bot.chatHistory = append(bot.chatHistory, assistantMessage)

	// Keep chat history manageable
	if len(bot.chatHistory) > 10 {
		bot.chatHistory = bot.chatHistory[len(bot.chatHistory)-10:]
	}

	return resp.Text, nil
}

// GetTumorInfo gets comprehensive information about a specific type of brain tumor
func (bot *BrainTumorBot) GetTumorInfo(ctx context.Context, tumorType string, location string, size float64) (string, error) {
	prompt := "Please provide detailed information about " + tumorType + " brain tumors"
	if location != "" {
		prompt += " in the " + location + " region"
	}
	if size > 0 {
		prompt += " of size " + string(size)
	}
	prompt += `. Include:
	1. Detailed description and characteristics
	2. Common symptoms and warning signs
	3. Treatment options and approaches
	4. Prognosis and survival rates
	5. Risk factors and prevention
	Keep the response medically accurate and cite sources where possible.`

	return bot.Chat(ctx, prompt, &TumorContext{
		Type:     tumorType,
		Location: location,
		Size:     size,
	})
}

// GetTreatmentOptions gets specific treatment options for a tumor type
func (bot *BrainTumorBot) GetTreatmentOptions(ctx context.Context, tumorType string) (string, error) {
	prompt := "As a medical expert, provide detailed treatment options for " + tumorType + ` brain tumors.
	Include:
	1. Surgical options and techniques
	2. Radiation therapy approaches
	3. Chemotherapy protocols
	4. Targeted therapies
	5. Clinical trials
	6. Rehabilitation and follow-up care
	Base your response on current medical research and guidelines.`

	return bot.Chat(ctx, prompt, &TumorContext{Type: tumorType})
}

// GetSymptoms gets symptoms based on tumor type and location
func (bot *BrainTumorBot) GetSymptoms(ctx context.Context, tumorType string, location string) (string, error) {
	prompt := "As a medical expert, describe the symptoms of " + tumorType + " brain tumors"
	if location != "" {
		prompt += " in the " + location + " region"
	}
	prompt += `:
	1. Common symptoms
	2. Location-specific symptoms
	3. Early warning signs
	4. Progressive symptoms
	5. Emergency symptoms
	Include both general and specific symptoms based on current medical literature.`

	return bot.Chat(ctx, prompt, &TumorContext{
		Type:     tumorType,
		Location: location,
	})
}

func main() {
	// Create a new bot instance
	bot := NewBrainTumorBot("Ff74gn0LI9T")

	// Example usage
	ctx := context.Background()

	// Get information about pituitary and glioma tumors
	response, err := bot.GetTumorInfo(ctx, "pituitary and glioma", "brain", 0)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Tumor Comparison:\n%s", response)

	// Example chat interaction
	chatResponse, err := bot.Chat(ctx, "What are the symptoms of my pituitary tumor?", &TumorContext{
		Type:     "pituitary",
		Location: "pituitary fossa",
		Size:     2.0,
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Chat Response:\n%s", chatResponse)
} 