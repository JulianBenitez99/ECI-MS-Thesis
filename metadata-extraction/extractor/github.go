package extractor

import (
	"encoding/json"
	"fmt"
	"github.com/PuerkitoBio/goquery"
	"github.com/go-git/go-git/v5"
	"metadata-extraction/model"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func GetRepositoryMetadata(repo *model.Repository) (*model.Metadata, error) {
	path, err := clone(repo)
	if err != nil {
		return nil, err
	}
	if path == "" {
		return nil, nil
	}
	languages, err := getLanguages(repo)
	if err != nil {
		return nil, err
	}
	commits, err := getCommits(path)
	if err != nil {
		return nil, err
	}
	contributors, err := getContributors(repo)
	if err != nil {
		return nil, err
	}
	defer func(path string) {
		err := deleteDir(path)
		if err != nil {
			fmt.Println(err)
		}
	}(path)

	return &model.Metadata{
		Owner:        repo.Owner,
		Name:         repo.Name,
		URL:          repo.URL,
		CreatedAt:    repo.CreatedAt,
		Stars:        repo.Stars,
		Issues:       repo.Issues,
		License:      repo.License,
		Languages:    languages,
		Commits:      commits,
		Contributors: contributors,
	}, nil
}

func clone(repo *model.Repository) (string, error) {
	path := fmt.Sprintf("/tmp/%s--%s", repo.Owner, repo.Name)
	_, err := git.PlainClone(path, false, &git.CloneOptions{
		URL:      repo.URL,
		Progress: os.Stdout,
	})
	if err != nil && strings.Contains(err.Error(), "authentication required") {
		fmt.Println(fmt.Sprintf("Skipping repository %s/%s due to authentication required error", repo.Owner, repo.Name))
		return "", nil
	}
	return path, err
}

// get total number of commits from the HEAD revision
func getCommits(path string) (int, error) {
	cmd := exec.Command("git", "rev-list", "--count", "HEAD")
	cmd.Dir = path
	outBytes, err := cmd.Output()
	if err != nil {
		return -1, err
	}
	rawOutput := strings.TrimSuffix(string(outBytes), "\n")
	atoi, err := strconv.Atoi(rawOutput)
	if err != nil {
		return -1, err
	}
	return atoi, nil
}

func getContributors(repo *model.Repository) (int, error) {
	res, err := http.Get(repo.URL)
	if err != nil {
		return -1, err
	}
	defer res.Body.Close()
	doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		return -1, err
	}
	contributorsRaw := doc.Find("div.BorderGrid-row:nth-child(4) > div:nth-child(1) > h2:nth-child(1) > a:nth-child(1) > span:nth-child(1)").
		Text()
	contributorsStr := strings.Replace(contributorsRaw, ",", "", -1)
	contributors, err := strconv.Atoi(contributorsStr)
	if err != nil {
		return -1, err
	}
	return contributors, nil
}

func getLanguages(repo *model.Repository) (map[string]int, error) {
	remaining := verifyRateLimit()
	fmt.Println("Remaining rate limit: " + fmt.Sprint(remaining))
	if remaining <= 0 {
		return nil, fmt.Errorf("rate limit exceeded")
	}
	cmd := exec.Command("gh", "api", "-H", "Accept: application/json",
		"-H", "X-GitHub-Api-Version: 2022-11-28", fmt.Sprintf("/repos/%s/%s/languages", repo.Owner, repo.Name))
	outBytes, err := cmd.Output()
	if err != nil {
		output := string(outBytes)
		if strings.Contains(output, "Not Found") || strings.Contains(output, "access blocked") {
			return nil, nil
		}
		return nil, err
	}
	var languages map[string]int
	err = json.Unmarshal(outBytes, &languages)
	return languages, err
}

func verifyRateLimit() int {
	cmd := exec.Command("gh", "api", "rate_limit", "-q", ".resources.core.remaining")
	output, err := cmd.Output()
	if err != nil {
		return -1
	}
	rawOutput := strings.TrimSuffix(string(output), "\n")
	atoi, err := strconv.Atoi(rawOutput)
	if err != nil {
		return -1
	}
	return atoi
}

func deleteDir(path string) error {
	if strings.HasPrefix(path, "/tmp") && path != "/tmp" {
		return fmt.Errorf("cannot delete %s", path)
	}
	return os.RemoveAll(path)
}
