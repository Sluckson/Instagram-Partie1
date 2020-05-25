package com.example.instagramclone;

import android.app.Application;

import com.parse.Parse;
import com.parse.ParseObject;

public class ParseApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();

        // Register your parse models
        ParseObject.registerSubclass(Post.class);

        Parse.initialize(new Parse.Configuration.Builder(this)
                .applicationId("surprice-parstagram") // should correspond to APP_ID env variable
                .clientKey("CodepathMovePath")  // set explicitly unless clientKey is explicitly configured on Parse server
                .server("https://surprice-parstagram.herokuapp.com/parse").build());
    }
}

